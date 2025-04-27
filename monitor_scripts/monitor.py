import argparse
import os
import json
import pprint
import csv
from collections import defaultdict, Counter
from concurrent.futures import ProcessPoolExecutor
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from consts import (
    GAMMAS,
    OFFICIAL_PASSED_TASK_IDS_PATH,
    MBPP_SIZE,
    DEEPSEEK_13B_INSTRUCT_MODEL_NAME,
    DEEPSEEK_13_SOLVED_TASK_IDS,
)


def load_json(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Failed loading {path}")
        return None


def load_official_passed_ids(json_path):
    try:
        with open(json_path, "r") as f:
            return set(json.load(f))
    except Exception as e:
        print(f"Error loading baseline passed IDs from {json_path}: {e}")
        return set()

    samples = defaultdict(dict)

    for fname in os.listdir(trial_dir):
        if not fname.startswith("task_id=") or not fname.endswith(".json"):
            continue

        try:
            parts = fname[len("task_id=") :].split("_gamma=")
            task_id = int(parts[0])
            gamma = float(parts[1][:-5])
        except Exception as e:
            print(f"Skipping file {fname} due to parsing error: {e}")
            continue

        data = load_json(os.path.join(trial_dir, fname))
        if data is None:
            continue

        samples[task_id][gamma] = data

    improved_ids = []
    total = 0
    total_clean = 0
    passed_clean = 0
    failed_samples = 0
    failed_with_error = 0

    for task_id, gamma_results in samples.items():
        base = gamma_results.get(0.0)
        if base is None:
            continue

        total += 1
        is_error = base.get("general_error") or base.get("has_testcase_error")
        if not is_error:
            total_clean += 1
            if base.get("passed"):
                passed_clean += 1

        if not base.get("passed"):
            failed_samples += 1
            if base.get("has_testcase_error") or base.get("general_error"):
                failed_with_error += 1

            for gamma, result in gamma_results.items():
                if gamma == 0.0:
                    continue
                if (
                    result.get("passed")
                    and not result.get("general_error")
                    and not result.get("has_testcase_error")
                ):
                    improved_ids.append(task_id)
                    break

    improved_ids = list(set(improved_ids))
    improvement_count = len(improved_ids)
    passed_without_improvement = passed_clean - improvement_count
    pass_rate = passed_clean / total_clean if total_clean else 0
    error_rate = failed_with_error / failed_samples if failed_samples else 0

    print(f"Trial directory: {trial_dir}")
    print(f"Total samples: {total}")
    print(f"Improved samples: {improvement_count}")
    print(f"Improvement percentage (over all): {improvement_count / total * 100:.2f}%")
    print(
        f"Improvement percentage (clean only): {improvement_count / total_clean * 100:.2f}%"
    )
    if failed_samples:
        print(f"Error rate among failed samples: {error_rate * 100:.2f}%")
    print(f"Improved sample IDs: {sorted(improved_ids)}")

    if generate_csv:
        task_ids = sorted(samples.keys())
        gamma_values = sorted(GAMMAS)

        accuracy_rows = []
        passed_rows = []

        header = ["task_id"] + [str(g) for g in gamma_values]

        for task_id in task_ids:
            acc_row = [str(task_id)]
            pass_row = [str(task_id)]
            for gamma in gamma_values:
                data = samples[task_id].get(gamma)
                if data is None:
                    acc_row.append("")
                    pass_row.append("")
                else:
                    acc = data.get("accuracy")
                    passed = data.get("passed")
                    acc_row.append(f"{acc:.4f}" if acc is not None else "")
                    pass_row.append("1" if passed else "0")
            accuracy_rows.append(acc_row)
            passed_rows.append(pass_row)

        with open(os.path.join(trial_dir, "accuracy.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(accuracy_rows)

        with open(os.path.join(trial_dir, "passed.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(passed_rows)

        print(f"CSV files saved to {trial_dir}[accuracy.csv, passed.csv]")

    return {
        "dir": trial_dir,
        "samples": samples,
        "improved_ids": improved_ids,
        "improvement_count": improvement_count,
        "pass_rate": pass_rate,
        "passed_clean": passed_clean,
        "passed_without_improvement": passed_without_improvement,
        "total_clean": total_clean,
        "error_rate": error_rate,
        "failed_samples": failed_samples,
        "failed_with_error": failed_with_error,
        "total_samples": total,
    }


def analyze_trial(trial_dir, generate_csv, model_name):
    invalid_samples = []

    samples = defaultdict(dict)
    baseline_passed_ids = load_official_passed_ids(
        OFFICIAL_PASSED_TASK_IDS_PATH[model_name]
    )

    for fname in os.listdir(trial_dir):
        if not fname.startswith("task_id=") or not fname.endswith(".json"):
            continue

        try:
            parts = fname[len("task_id=") :].split("_gamma=")
            task_id = int(parts[0])
            gamma = float(parts[1][:-5])
        except Exception as e:
            print(f"Skipping file {fname} due to parsing error: {e}")
            continue

        data = load_json(os.path.join(trial_dir, fname))
        if data is None:
            invalid_samples.append(os.path.join(trial_dir, fname))
            continue

        samples[task_id][gamma] = data

    improved_ids = set()
    passed_baseline = set()
    passed_total = set()

    total = 0
    total_clean = 0
    failed_samples = 0
    failed_with_error = 0

    for task_id, gamma_results in samples.items():
        base = gamma_results.get(0.0)
        if base is None:
            continue

        total += 1
        is_error = base.get("general_error") or base.get("has_testcase_error")
        if not is_error:
            total_clean += 1

        # Track baseline pass
        if base.get("passed"):
            if task_id in baseline_passed_ids:
                passed_baseline.add(task_id)
                passed_total.add(task_id)

        # Check for improvement via guided (gamma > 0)
        for gamma, result in gamma_results.items():
            if gamma == 0.0:
                continue
            if (
                result.get("passed")
                and not result.get("general_error")
                and not result.get("has_testcase_error")
            ):
                if task_id not in baseline_passed_ids:
                    improved_ids.add(task_id)
                passed_total.add(task_id)
                # if model_name == DEEPSEEK_13B_INSTRUCT_MODEL_NAME:
                # if task_id in DEEPSEEK_13_SOLVED_TASK_IDS:
                # pass
                # improved_ids.add(task_id)
                # passed_total.add(task_id)
                break  # stop at first successful gamma > 0

        # Track error rate on failed samples at gamma=0
        if not base.get("passed"):
            failed_samples += 1
            if base.get("has_testcase_error") or base.get("general_error"):
                failed_with_error += 1

    improvement_count = len(improved_ids)
    passed_without_improvement = len(passed_baseline)
    total_passed = len(passed_total)
    error_rate = failed_with_error / failed_samples if failed_samples else 0
    if not total_clean:
        total_clean += 1e-8

    print(f"Trial directory: {trial_dir}")
    print(f"Total samples: {total}")
    print(f"Improved samples: {improvement_count}")
    print(f"Passed without improvement (baseline passed): {passed_without_improvement}")
    print(f"Total passed (any gamma): {total_passed}")
    print(
        f"Improvement percentage (clean only): {improvement_count / total_clean * 100:.2f}%"
    )
    if failed_samples:
        print(f"Error rate among failed samples: {error_rate * 100:.2f}%")
    print(f"Improved sample IDs: {sorted(improved_ids)}")
    # pprint.pprint(f"Invalid Samples:\n{invalid_samples}")

    # CSV logic here if needed
    if generate_csv:
        task_ids = sorted(samples.keys())
        gamma_values = sorted(GAMMAS)

        accuracy_rows = []
        passed_rows = []
        header = ["task_id"] + [str(g) for g in gamma_values]

        for task_id in task_ids:
            acc_row = [str(task_id)]
            pass_row = [str(task_id)]
            for gamma in gamma_values:
                data = samples[task_id].get(gamma)
                if data is None:
                    acc_row.append("")
                    pass_row.append("")
                else:
                    acc = data.get("accuracy")
                    passed = data.get("passed")
                    acc_row.append(f"{acc:.4f}" if acc is not None else "")
                    pass_row.append("1" if passed else "0")
            accuracy_rows.append(acc_row)
            passed_rows.append(pass_row)

        with open(os.path.join(trial_dir, "accuracy.csv"), "w", newline="") as f:
            csv.writer(f).writerows([header] + accuracy_rows)

        with open(os.path.join(trial_dir, "passed.csv"), "w", newline="") as f:
            csv.writer(f).writerows([header] + passed_rows)

        print(f"CSV files saved to {trial_dir}/accuracy.csv and passed.csv")
        print()

    return {
        "dir": trial_dir,
        "samples": samples,
        "improved_ids": sorted(improved_ids),
        "improvement_count": improvement_count,
        "passed_clean": passed_without_improvement,
        "passed_without_improvement": passed_without_improvement,
        "passed_total": total_passed,
        "total_clean": total_clean,
        "error_rate": error_rate,
        "failed_samples": failed_samples,
        "failed_with_error": failed_with_error,
        "total_samples": total,
    }


def analyze_trial_wrapper(args):
    trial_dir, generate_csv, model_name = args
    return analyze_trial(trial_dir, generate_csv, model_name)


def aggregate_analysis(base_dir, model_name):
    official_passed_ids = load_official_passed_ids(
        OFFICIAL_PASSED_TASK_IDS_PATH[model_name]
    )
    trial_results = {}
    base_passed = defaultdict(bool)

    trial_dirs = [
        os.path.join(base_dir, subdir)
        for subdir in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, subdir)) and not subdir.startswith('.')
    ]

    with ProcessPoolExecutor() as executor:
        results = list(
            executor.map(
                analyze_trial_wrapper,
                [(trial_dir, True, model_name) for trial_dir in trial_dirs],
            )
        )

    for trial_dir, result in zip(trial_dirs, results):
        subdir = os.path.basename(trial_dir)
        trial_results[subdir] = result
        for task_id, gamma_map in result["samples"].items():
            if gamma_map.get(0.0, {}).get("passed"):
                base_passed[task_id] = True

    all_improved_counter = Counter()
    improved_task_to_trials = defaultdict(list)
    complete_scores = []
    incomplete_scores = []
    complete_improved_ids = set()

    print("\n===== AGGREGATE ANALYSIS =====")

    for trial_name, result in trial_results.items():
        improved_ids = set(result["improved_ids"])
        passed_wo_impr_ids = set(
            tid
            for tid in result["samples"]
            if tid in official_passed_ids
            and result["samples"][tid].get(0.0, {}).get("passed")
        )

        filtered_improved = sorted(improved_ids - official_passed_ids)

        for tid in filtered_improved:
            improved_task_to_trials[tid].append(trial_name)
        all_improved_counter.update(filtered_improved)
        if not result["total_samples"]:
            result["total_samples"] = 1e-8

        score = (
            len(filtered_improved),  # improved count
            result["passed_total"] / result["total_samples"],  # pass rate
            result["passed_total"],  # passed
            len(passed_wo_impr_ids),  # passed w/o improvement
            result["total_clean"],
            result["error_rate"],
            result["failed_with_error"],
            result["failed_samples"],
            trial_name,
            result["total_samples"],
            set(filtered_improved),
        )

        if result["total_samples"] >= 0.95 * MBPP_SIZE:
            complete_scores.append(score)
        else:
            incomplete_scores.append(score)

    print("Improved sample IDs with frequency and trials (sorted by frequency):")
    for task_id, count in sorted(
        all_improved_counter.items(), key=lambda x: (-x[1], x[0])
    ):
        trial_names = ", ".join(sorted(improved_task_to_trials[task_id]))
        print(f"  task_id={task_id}: {count} trials | {trial_names}")

    complete_scores.sort(key=lambda x: x[0], reverse=True)
    incomplete_scores.sort(key=lambda x: x[0], reverse=True)

    print("\n===== TRIAL RANKING BY PASS RATE =====")
    print("\n--- Complete Trials (>= 95% MBPP samples) ---")
    grid_search_entries = []
    cumulative = set()
    for (
        improved,
        pass_rate,
        passed,
        passed_wo_impr,
        total,
        err_rate,
        err_count,
        fail_count,
        name,
        total_samples,
        improved_ids,
    ) in complete_scores:
        cumulative |= improved_ids
        print(
            f"{name}: {pass_rate*100:.2f}% | improved: ({improved}/{total}) {improved/total*100:.2f}% | passed: ({passed}/{total}) {passed/total*100:.2f}% | passed w/o improvement: ({passed_wo_impr}/{total}) {passed_wo_impr/total*100:.2f}% | error%: {err_rate*100:.2f}% ({err_count}/{fail_count}) | Cimp: {len(cumulative)} / {MBPP_SIZE} = {len(cumulative)/MBPP_SIZE*100:.2f}%"
        )
        cumulative_list = list(improved_ids)
        cumulative_list.sort()
        grid_search_entry = {"name": name, "task_ids": cumulative_list}
        grid_search_entries.append(grid_search_entry)

    cumulative_path = os.path.join(base_dir, "cumulative.json")
    with open(cumulative_path, "w") as f:
        json.dump(grid_search_entries, f)

    print("\n--- Incomplete Trials (< 95% MBPP samples) ---")
    for (
        improved,
        pass_rate,
        passed,
        passed_wo_impr,
        total,
        err_rate,
        err_count,
        fail_count,
        name,
        total_samples,
        _,
    ) in incomplete_scores:
        print(
            f"{name}: {pass_rate*100:.2f}% | improved: ({improved}/{total}) {improved/total*100:.2f}% | passed: ({passed}/{total}) {passed/total*100:.2f}% | passed w/o improvement: ({passed_wo_impr}/{total}) {passed_wo_impr/total*100:.2f}% | error%: {err_rate*100:.2f}% ({err_count}/{fail_count})"
        )

    missing_base_ids = sorted(
        tid for tid in official_passed_ids if not base_passed[tid]
    )
    print("\n===== BASELINE-PASSED BUT NEVER SEEN PASSED IN TRIALS (gamma=0.0) =====")
    print(f"Count: {len(missing_base_ids)}")
    print(f"task_ids: {missing_base_ids}")

    print("\n===== FINAL RESULTS =====")
    print(f"Total improved samples across all trials: {len(all_improved_counter)}")
    print(
        f"Improvement percentage over MBPP: {len(all_improved_counter) / MBPP_SIZE * 100:.2f}%"
    )
    total_wo = 257
    if "V2" in model_name:
        total_wo = 307
    total_passed_counter = total_wo + len(all_improved_counter)
    print(f"Passed over MBPP (Count): {total_passed_counter} / {MBPP_SIZE}")
    print(f"Passed over MBPP: {total_passed_counter / MBPP_SIZE * 100:.2f}%")
    print()
    all_improved_counter = list(all_improved_counter)
    all_improved_counter.sort()
    print(all_improved_counter)
    print(f"{len(all_improved_counter)}")
    print(cumulative_path)


def main():
    parser = argparse.ArgumentParser(
        description="Monitor MBPP guided inference results."
    )
    parser.add_argument(
        "--trial-dir", type=str, help="Path to a single trial directory to analyze"
    )
    parser.add_argument(
        "--aggregate-dir",
        type=str,
        help="Path to parent dir of multiple trials for aggregation",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="If set, generate CSVs (only with --trial-dir)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEEPSEEK_13B_INSTRUCT_MODEL_NAME,
        help="Model name for display/logging purposes (default: %(default)s)",
    )

    args = parser.parse_args()

    if args.trial_dir:
        analyze_trial(args.trial_dir, args.csv, args.model_name)
    elif args.aggregate_dir:
        aggregate_analysis(args.aggregate_dir, model_name=args.model_name)
    else:
        print("Please provide either --trial-dir or --aggregate-dir")


if __name__ == "__main__":
    main()
