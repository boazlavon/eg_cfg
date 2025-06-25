import os
import json
import re
import traceback
import argparse
import random
from typing import List, Tuple, Dict

from datasets import load_dataset
from collections import OrderedDict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from datasets_utils import load_codecontests_problems
from eval_utils import (
    run_tests,
    exec_eval__run_tests,
    load_humaneval_et_problems,
    load_mbpp_et_problems,
)
from eg_cfg.eg_cfg_session_manager import format_results
from eg_cfg.api_comm import APICommunication

EXEC_EVAL_DEFAULT_HOST_PORT = 5000
EVAL_TYPE__EG_CFG = "eval-eg-cfg"
EVAL_TYPE__EXEC_EVAL = "ExecEval"
AVAILABLE_EVAL_TYPES = (EVAL_TYPE__EG_CFG, EVAL_TYPE__EXEC_EVAL)

DATASET__MBPP_ET = "mbpp-et"
DATASET__HUMANEVAL_ET = "humaneval-et"
DATASET__CODECONTESTS_HF = "codecontests-hf"
AVAILABLE_DATASETS = (
    DATASET__MBPP_ET,
    DATASET__HUMANEVAL_ET,
    DATASET__CODECONTESTS_HF,
)
EVAL_DEFAULT_WORKERS = 4  # Default number of workers per trial


# -------------------- Evaluation Job -------------------- #
def process_file(
    trial_path,
    output_dir,
    task_id,
    filename,
    test_cases,
    inputs,
    eval_type,
    io_flag,
    api_comm,
):
    input_file = trial_path / filename
    output_file = output_dir / filename

    if output_file.exists():
        # print(f"Skipping (exists): {output_file}")
        return

    try:
        with open(input_file, "r") as f:
            data = json.load(f)
            code = data.get("code")
            if not code:
                # print(f"No code in {filename}")
                return

        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            f.write("")

        print(f"Evaluating {filename}")
        general_error = ""
        tb = None
        try:
            solution_entry = None
            if eval_type == EVAL_TYPE__EG_CFG:
                solution_results = run_tests(code, test_cases, inputs, io_flag=io_flag)
            elif eval_type == EVAL_TYPE__EXEC_EVAL:
                solution_entry = exec_eval__run_tests(
                    code, test_cases, task_id, api_comm
                )
                # solution_entry['test_list'] = dict_test_cases
        except Exception as e:
            general_error = str(e)
            tb = traceback.format_exc()

        if solution_entry is None:
            solution_entry = format_results(code, solution_results, general_error, tb)

        solution_entry["general_error"] = general_error
        solution_entry["tb"] = tb

        with open(output_file, "w") as f:
            json.dump(solution_entry, f, indent=2)
        print(f"Saved: {output_file}")

    except Exception as e:
        print(f"Failed to process {filename}: {e}")


# -------------------- Main -------------------- #
def main(
    root_dir,
    output_base,
    dataset,
    eval_type,
    trial_workers=EVAL_DEFAULT_WORKERS,
    exec_eval_host_ip=None,
    exec_eval_host_port=None,
):
    if dataset == DATASET__MBPP_ET:
        problems = load_mbpp_et_problems()
    elif dataset == DATASET__HUMANEVAL_ET:
        problems = load_humaneval_et_problems()
    elif dataset == DATASET__CODECONTESTS_HF:
        problems = load_codecontests_problems()
    print(f"Loaded {dataset} dataset ({len(problems)} problems)")

    root_dir = Path(root_dir)
    output_base = Path(output_base)

    trial_dirs = os.listdir(root_dir)
    random.shuffle(trial_dirs)
    for trial_dir in trial_dirs:
        trial_path = root_dir / trial_dir
        if not trial_path.is_dir():
            continue
        if trial_dir.startswith("."):
            continue

        output_dir = output_base / trial_dir
        filenames = [f for f in os.listdir(trial_path) if f.endswith(".json")]

        api_comm = None
        if eval_type == EVAL_TYPE__EXEC_EVAL:
            assert (
                exec_eval_host_ip is not None
            ), "ExecEval host IP must be provided for ExecEval evaluation"
            assert (
                exec_eval_host_port is not None
            ), "ExecEval host port must be provided for ExecEval evaluation"
            api_comm = APICommunication(
                f"http://{exec_eval_host_ip}:{exec_eval_host_port}"
            )
        jobs = []
        if not filenames:
            continue
        for filename in filenames:
            if dataset in (DATASET__MBPP_ET,):
                match = re.search(r"task_id=(\d+)", filename)
                task_id = int(match.group(1))
            elif dataset in (DATASET__CODECONTESTS_HF,):
                match = re.search(r"task_id=([0-9]+_[A-Z])", filename)
                task_id = match.group(1)
            elif dataset in (DATASET__HUMANEVAL_ET,):
                match = re.search(r"task_id=HumanEval_(\d+)", filename)
                task_id = int(match.group(1))
            if not match:
                continue
            if dataset in (DATASET__HUMANEVAL_ET,):
                task_id = f"HumanEval/{task_id}"
            if task_id not in problems:
                continue

            if dataset in (DATASET__CODECONTESTS_HF,):
                eval_test_cases = problems[task_id]["eval_test_list"]
            else:
                eval_test_cases = problems[task_id]["test_list"]
            inputs = None
            io_flag = dataset in (DATASET__CODECONTESTS_HF,)
            jobs.append(
                (
                    trial_path,
                    output_dir,
                    task_id,
                    filename,
                    eval_test_cases,
                    inputs,
                    eval_type,
                    io_flag,
                    api_comm,
                )
            )

        # print()
        # print(f"Trial {trial_dir}: Dispatching {len(jobs)} evaluations with {trial_workers} threads")
        # with ThreadPoolExecutor(max_workers=trial_workers) as executor:
        #     futures = [executor.submit(process_file, *job) for job in jobs]
        #     for f in as_completed(futures):
        #         f.result()  # raise exceptions if any
        # FOR DEBUGGING
        for job in jobs:
            process_file(*job)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "source_trials_dir", help="Directory with trial subfolders (Source Trials)"
    )
    parser.add_argument(
        "--output-dir",
        default="processed_trials",
        help="Directory to write output trials",
    )
    parser.add_argument(
        "--trial-workers",
        type=int,
        default=EVAL_DEFAULT_WORKERS,
        help="Number of threads per trial",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=AVAILABLE_DATASETS,
        required=True,
        help="Dataset. Options: %(choices)s",
    )
    parser.add_argument(
        "--eval-type",
        type=str,
        choices=AVAILABLE_EVAL_TYPES,
        default=EVAL_TYPE__EG_CFG,
        required=False,
        help="Evaluation type. Options: %(choices)s",
    )
    parser.add_argument(
        "--exec-eval-host-ip",
        type=str,
        default=None,
        help="IP address for execution evaluation host",
    )
    parser.add_argument(
        "--exec-eval-host-port",
        type=int,
        default=EXEC_EVAL_DEFAULT_HOST_PORT,
        help="Port for execution evaluation host",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        args.source_trials_dir,
        args.output_dir,
        args.dataset,
        args.eval_type,
        args.trial_workers,
        args.exec_eval_host_ip,
        args.exec_eval_host_port,
    )
