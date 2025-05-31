import os
import json
import re
import time
import traceback
import subprocess
import tempfile
import argparse
import random

from datasets import load_dataset
from collections import OrderedDict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# -------------------- Global Evaluation Cache -------------------- #
evaluation_cache = {}


# -------------------- Test Execution -------------------- #
def run_tests(solution, test_cases):
    results = {}
    for test_case in test_cases:
        if solution is None:
            results[test_case] = {
                "result": False,
                "time": -1,
                "error": "GenerationError",
            }
            continue

        try:
            results[test_case] = evaluate_solution(solution, test_case)
        except Exception as e:
            tb = traceback.format_exc()
            results[test_case] = {
                "result": False,
                "time": -1,
                "error": str(type(e)),
                "tb": tb,
            }
            print(f"Problem executing test case: {test_case}")
    return results


def evaluate_solution(code, test_case, timeout=10):
    key = (code, test_case)
    if key in evaluation_cache:
        return evaluation_cache[key]

    test_passed = False
    error = None
    test_code = f"{code}\n{test_case}"

    with tempfile.NamedTemporaryFile(suffix=".py", mode="w") as temp_file:
        temp_file.write(test_code)
        temp_file.flush()

        start_time = time.time()
        try:
            result = subprocess.run(
                ["python", temp_file.name],
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            if result.returncode == 0 and "Traceback" not in result.stderr:
                test_passed = True

        except subprocess.TimeoutExpired:
            error = "Timeout"
        except Exception:
            error = "Exception"
        finally:
            delta_time = time.time() - start_time

    result_entry = {
        "result": test_passed,
        "time": delta_time,
        "error": error,
    }
    if error is None:
        evaluation_cache[key] = result_entry
    return result_entry


def format_results(solution, results, general_error, tb=None):
    passed = all(r["result"] for r in results.values())
    correct = sum(int(r["result"]) for r in results.values())
    total = len(results)
    accuracy = correct / total if total else 0.0
    has_testcase_error = all([bool(result["error"]) for result in results.values()])
    entry = {
        "code": solution,
        "results": results,
        "passed": passed,
        "accuracy": accuracy,
        "general_error": general_error,
        "has_testcase_error": has_testcase_error,
    }
    if tb is not None:
        entry["tb"] = tb
    return entry


# -------------------- MBPP-ET Loader -------------------- #
def load_mbpp_et_problems():
    test_ds = load_dataset("dz1/CodeScore-MBPP-ET", split="train")
    problems = OrderedDict((example["task_id"], example) for example in test_ds)
    return problems


# -------------------- Evaluation Job -------------------- #
def process_file(trial_path, output_dir, filename, test_cases):
    input_file = trial_path / filename
    output_file = output_dir / filename

    if output_file.exists():
        print(f"⏩ Skipping (exists): {output_file}")
        return

    try:
        with open(input_file, "r") as f:
            data = json.load(f)
            code = data.get("code")
            if code is None:
                print(f"⚠️  No code in {filename}")
                return

        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            f.write("{}")  # reserve the slot

        print(f"🧪 Evaluating {filename}")
        general_error = ""
        tb = None
        solution_results = run_tests(code, test_cases)
        solution_entry = format_results(code, solution_results, general_error, tb)

        with open(output_file, "w") as f:
            json.dump(solution_entry, f, indent=2)
        print(f"📎 Saved: {output_file}")

    except Exception as e:
        print(f"❌ Failed to process {filename}: {e}")


# -------------------- Main -------------------- #
def main(root_dir, output_base, trial_workers=4):
    problems = load_mbpp_et_problems()
    print(f"✅ Loaded {len(problems)} MBPP-ET problems")

    root_dir = Path(root_dir)
    output_base = Path(output_base)


    trial_dirs = os.listdir(root_dir)
    random.shuffle(trial_dirs)
    for trial_dir in trial_dirs:
        trial_path = root_dir / trial_dir
        if not trial_path.is_dir():
            continue

        output_dir = output_base / trial_dir
        filenames = [f for f in os.listdir(trial_path) if f.endswith(".json")]

        jobs = []
        if not filenames:
            continue
        for filename in filenames:
            match = re.search(r"task_id=(\d+)", filename)
            if not match:
                continue
            task_id = int(match.group(1))

            if task_id not in problems:
                continue

            test_cases = problems[task_id]["test_list"]
            jobs.append((trial_path, output_dir, filename, test_cases))

        print(f"\n📁 Trial {trial_dir}: Dispatching {len(jobs)} evaluations with {trial_workers} threads")

        with ThreadPoolExecutor(max_workers=trial_workers) as executor:
            futures = [executor.submit(process_file, *job) for job in jobs]
            for f in as_completed(futures):
                f.result()  # raise exceptions if any


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root_dir", help="Root directory with trial subfolders")
    parser.add_argument("--output-dir", default="processed_trials", help="Directory to write updated outputs")
    parser.add_argument("--trial-workers", type=int, default=4, help="Number of threads per trial")
    args = parser.parse_args()
    main(args.root_dir, args.output_dir, args.trial_workers)
