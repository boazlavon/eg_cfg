import os
import json
import re
import traceback
import argparse
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from datasets_utils import LOAD_DATASET_HANDLER
from eval_utils import run_tests
from exec_eval_utils import exec_eval__run_tests
from eg_cfg_session_manager import format_results
from exec_eval_utils import ExecEval__APICommunication
from consts import *


def process_file(
    trial_path,
    output_dir,
    filename,
    test_cases,
    inputs,
    eval_type,
    io_flag,
    exec_eval_session,
):
    input_file = trial_path / filename
    output_file = output_dir / filename

    if output_file.exists():
        return

    try:
        with open(input_file, "r") as f:
            data = json.load(f)
            code = data.get("code")
            if not code:
                return

        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            f.write("")

        print(f"Evaluating {filename}")
        general_error = None
        tb = None
        solution_entry = None
        try:
            if eval_type == EVAL_TYPE__EG_CFG:
                solution_results = run_tests(code, test_cases, io_flag=io_flag)
                solution_entry = format_results(
                    code, solution_results, general_error, tb
                )
            elif eval_type == EVAL_TYPE__EXEC_EVAL:
                solution_entry = exec_eval__run_tests(
                    code, test_cases, exec_eval_session
                )
        except Exception as e:
            general_error = str(e)
            tb = traceback.format_exc()

        if solution_entry is not None:
            solution_entry["general_error"] = general_error
            solution_entry["tb"] = tb

        with open(output_file, "w") as f:
            json.dump(solution_entry, f, indent=2)
        print(f"Saved: {output_file}")

    except Exception as e:
        print(f"Failed to process {filename}: {e}")


def extract_task_id(filename, dataset):
    task_id = None
    try:
        if dataset in (DATASET__MBPP_ET,):
            match = re.search(r"task_id=(\d+)", filename)
            task_id = int(match.group(1)) if match else None
        elif dataset in (DATASET__CODECONTESTS,):
            match = re.search(r"task_id=([0-9]+_[A-Z])", filename)
            task_id = match.group(1) if match else None
        elif dataset in (DATASET__HUMANEVAL_ET,):
            match = re.search(r"task_id=HumanEval_(\d+)", filename)
            task_id = f"HumanEval/{int(match.group(1))}" if match else None
    except Exception as e:
        print(f"Error extracting task_id from {filename}: {e}")
        task_id = None
    return task_id


def eval_trial(
    trial_dir,
    source_dir,
    output_base,
    dataset,
    eval_type,
    problems,
    exec_eval_session=None,
    trial_workers=EVAL_DEFAULT_WORKERS,
):
    assert problems, "Problems must be provided for evaluation"
    if eval_type == EVAL_TYPE__EXEC_EVAL:
        assert (
            exec_eval_session is not None
        ), "Session object must be provided for ExecEval evaluation"

    trial_path = source_dir / trial_dir
    if not trial_path.is_dir():
        return
    if trial_dir.startswith("."):
        return

    output_dir = output_base / trial_dir
    results_filenames = [f for f in os.listdir(trial_path) if f.endswith(".json")]
    jobs = []

    for result_filename in results_filenames:
        task_id = extract_task_id(result_filename, dataset)
        if task_id is None:
            print(f"Skipping {result_filename}: No task_id found")
            continue
        if task_id not in problems:
            continue

        io_flag = dataset in (DATASET__CODECONTESTS,)
        if dataset in (DATASET__CODECONTESTS,):
            eval_test_cases = problems[task_id]["eval_test_list"]
        else:
            eval_test_cases = problems[task_id]["test_list"]
        inputs = None
        jobs.append(
            (
                trial_path,
                output_dir,
                result_filename,
                eval_test_cases,
                inputs,
                eval_type,
                io_flag,
                exec_eval_session,
            )
        )

    print()
    print(
        f"Trial {trial_dir}: Dispatching {len(jobs)} evaluations with {trial_workers} threads"
    )
    with ThreadPoolExecutor(max_workers=trial_workers) as executor:
        futures = [executor.submit(process_file, *job) for job in jobs]
        for f in as_completed(futures):
            f.result()  # raise exceptions if any
    # FOR DEBUGGING
    # for job in jobs:
    #     process_file(*job)


def main(
    source_dir,
    output_base,
    dataset,
    eval_type,
    exec_eval_host_ip=None,
    exec_eval_host_port=None,
):
    assert os.path.exists(source_dir), f"Source directory {source_dir} does not exist"
    assert os.path.exists(output_base), f"Output directory {output_base} does not exist"
    assert (
        dataset in AVAILABLE_EVAL_DATASETS
    ), f"Dataset must be one of {AVAILABLE_EVAL_DATASETS}"
    assert (
        eval_type in AVAILABLE_EVAL_TYPES
    ), f"Eval type must be one of {AVAILABLE_EVAL_TYPES}"
    if eval_type == EVAL_TYPE__EXEC_EVAL:
        assert (
            exec_eval_host_ip is not None
        ), "ExecEval host IP must be provided for ExecEval evaluation"
        assert (
            exec_eval_host_port is not None
        ), "ExecEval host port must be provided for ExecEval evaluation"

    source_dir = Path(source_dir)
    output_base = Path(output_base)
    trial_dirs = os.listdir(source_dir)
    random.shuffle(trial_dirs)

    handler = LOAD_DATASET_HANDLER.get(dataset)
    problems = handler() if handler else {}
    assert problems, f"No problems loaded for dataset {dataset}"
    print(f"Loaded {dataset} dataset ({len(problems)} problems)")

    exec_eval_session = None
    if eval_type == EVAL_TYPE__EXEC_EVAL:
        exec_eval_session = ExecEval__APICommunication(
            exec_eval_host_ip, exec_eval_host_port
        )

    for trial_dir in trial_dirs:
        eval_trial(
            trial_dir,
            source_dir,
            output_base,
            dataset,
            eval_type,
            problems,
            exec_eval_session,
        )


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
        "--dataset",
        type=str,
        choices=AVAILABLE_EVAL_DATASETS,
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
        args.exec_eval_host_ip,
        args.exec_eval_host_port,
    )
