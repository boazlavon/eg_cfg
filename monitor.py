import argparse
import os
import json
from collections import defaultdict, Counter

MBPP_SIZE = 500

def load_json(path):
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Failed to load {path}: {e}")
        return None

def analyze_trial(trial_dir):
    samples = defaultdict(dict)

    for fname in os.listdir(trial_dir):
        if not fname.startswith("task_id=") or not fname.endswith(".json"):
            continue

        try:
            parts = fname[len("task_id="):].split("_gamma=")
            task_id = int(parts[0])
            gamma = float(parts[1][:-5])  # remove .json
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
                if result.get("passed") and not result.get("general_error") and not result.get("has_testcase_error"):
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
    print(f"Improvement percentage (clean only): {improvement_count / total_clean * 100:.2f}%")
    if failed_samples:
        print(f"Error rate among failed samples: {error_rate * 100:.2f}%")
    print(f"Improved sample IDs: {sorted(improved_ids)}")

    return {
        "dir": trial_dir,
        "improved_ids": improved_ids,
        "improvement_count": improvement_count,
        "pass_rate": pass_rate,
        "passed_clean": passed_clean,
        "passed_without_improvement": passed_without_improvement,
        "total_clean": total_clean,
        "error_rate": error_rate,
        "failed_samples": failed_samples,
        "failed_with_error": failed_with_error,
        "total_samples": total
    }

def aggregate_analysis(base_dir):
    all_improved_counter = Counter()
    improved_task_to_trials = defaultdict(list)
    complete_scores = []
    incomplete_scores = []

    for subdir in os.listdir(base_dir):
        path = os.path.join(base_dir, subdir)
        if os.path.isdir(path):
            result = analyze_trial(path)
            for tid in result["improved_ids"]:
                improved_task_to_trials[tid].append(subdir)
            all_improved_counter.update(result["improved_ids"])
            score = (
                result["improvement_count"],
                result["pass_rate"],
                result["passed_clean"],
                result["passed_without_improvement"],
                result["total_clean"],
                result["error_rate"],
                result["failed_with_error"],
                result["failed_samples"],
                subdir,
                result["total_samples"]
            )
            if result["total_samples"] >= 0.95 * MBPP_SIZE:
                complete_scores.append(score)
            else:
                incomplete_scores.append(score)

    print("\n===== AGGREGATE ANALYSIS =====")
    print("Improved sample IDs with frequency and trials (sorted by frequency):")
    for task_id, count in sorted(all_improved_counter.items(), key=lambda x: (-x[1], x[0])):
        trial_names = ", ".join(sorted(improved_task_to_trials[task_id]))
        print(f"  task_id={task_id}: {count} trials | {trial_names}")
    print(f"Total improved samples across all trials: {len(all_improved_counter)}")
    print(f"Improvement percentage over MBPP: {len(all_improved_counter) / MBPP_SIZE * 100:.2f}%")

    complete_scores.sort(key=lambda x: x[0], reverse=True)
    incomplete_scores.sort(key=lambda x: x[0], reverse=True)

    print("\n===== TRIAL RANKING BY PASS RATE =====")
    print("\n--- Complete Trials (>= 95% MBPP samples) ---")
    for improved, pass_rate, passed, passed_wo_impr, total, err_rate, err_count, fail_count, name, total_samples in complete_scores:
        print(f"{name}: {pass_rate*100:.2f}% | improved: ({improved}/{total}) {improved/total*100:.2f}% | passed: ({passed}/{total}) {passed/total*100:.2f}% | passed w/o improvement: ({passed_wo_impr}/{total}) {passed_wo_impr/total*100:.2f}% | error%: {err_rate*100:.2f}% ({err_count}/{fail_count})")

    print("\n--- Incomplete Trials (< 95% MBPP samples) ---")
    for improved, pass_rate, passed, passed_wo_impr, total, err_rate, err_count, fail_count, name, total_samples in incomplete_scores:
        print(f"{name}: {pass_rate*100:.2f}% | improved: ({improved}/{total}) {improved/total*100:.2f}% | passed: ({passed}/{total}) {passed/total*100:.2f}% | passed w/o improvement: ({passed_wo_impr}/{total}) {passed_wo_impr/total*100:.2f}% | error%: {err_rate*100:.2f}% ({err_count}/{fail_count})")

def main():
    parser = argparse.ArgumentParser(description="Monitor MBPP guided inference results.")
    parser.add_argument('--trial-dir', type=str, help="Path to a single trial directory to analyze")
    parser.add_argument('--aggregate-dir', type=str, help="Path to parent dir of multiple trials for aggregation")

    args = parser.parse_args()

    if args.trial_dir:
        analyze_trial(args.trial_dir)
    elif args.aggregate_dir:
        aggregate_analysis(args.aggregate_dir)
    else:
        print("Please provide either --trial-dir or --aggregate-dir")

if __name__ == '__main__':
    main()
