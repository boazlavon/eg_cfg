import argparse
import os
import json
from collections import defaultdict
from pathlib import Path


def analyze_task_across_trials(root_dir, task_id):
    task_key = f"task_id={task_id}"
    code_to_sources = defaultdict(list)
    total_trials = 0

    print(f"🔍 Scanning root directory: {root_dir}")
    for trial_dir in os.listdir(root_dir):
        trial_path = os.path.join(root_dir, trial_dir)
        if not os.path.isdir(trial_path):
            continue

        #print(f"\n📁 Trial: {trial_dir}")
        matched_files = 0

        for filename in os.listdir(trial_path):
            if not filename.endswith(".json"):
                continue
            if task_key not in filename:
                continue
            if "_gamma=0.0" in filename:
                print(f"  ⏩ Skipping baseline file: {filename}")
                continue  # Skip unguided

            matched_files += 1
            filepath = os.path.join(trial_path, filename)
            print(f"  ✅ Found: {filename}")

            try:
                with open(filepath, "r") as f:
                    data = json.load(f)
                    code = data.get("code")
                    if code:
                        code_to_sources[code].append((trial_dir, filename))
                        total_trials += 1
                        print(f"    📥 Loaded code (len={len(code)} chars)")
                    else:
                        print(f"    ⚠️  No 'code' field in: {filename}")
            except Exception as e:
                print(f"    ❌ Failed to read {filename}: {e}")

        if matched_files == 0:
            print("  ⚠️  No matching JSON files for this trial.")

    print(f"\n✅ Total trials matched for task_id={task_id}: {total_trials}")
    print(f"🧬 Unique code variants found: {len(code_to_sources)}")
    return code_to_sources, total_trials


# def save_outputs(code_to_sources, total_trials, task_id, output_base="."):
#     output_dir = Path(output_base) / f"{task_id}_solutions"
#     output_dir.mkdir(parents=True, exist_ok=True)

#     # Save code → number of trials it appeared in
#     code_freq = {
#         code: len(sources)
#         for code, sources in code_to_sources.items()
#     }
#     full_code_freq = {
#         code: len(sources)
#         for code, sources in code_to_sources.items()
#     }
#     freq_path = output_dir / "code_frequencies.json"
#     with open(freq_path, "w") as f:
#         json.dump(code_freq, f, indent=2)
#     print(f"\n📊 Frequencies saved to: {freq_path}")

#     # Save individual solutions ordered by popularity
#     sorted_codes = sorted(code_to_sources.items(), key=lambda x: len(x[1]), reverse=True)
#     for i, (code, sources) in enumerate(sorted_codes, start=1):
#         code_path = output_dir / f"{task_id}__#{i}.py"
#         with open(code_path, "w") as f:
#             f.write(code)
#         print(f"  💾 Saved solution #{i} ({len(sources)} matches) → {code_path.name}")

#     print(f"\n🎉 Done. Solutions written to: {output_dir}")
def save_outputs(code_to_sources, total_trials, task_id, output_base="."):
    output_dir = Path(output_base) / f"{task_id}_solutions"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save code → frequency
    code_freq = {
        code: len(sources)
        for code, sources in code_to_sources.items()
    }
    freq_path = output_dir / "code_frequencies.json"
    with open(freq_path, "w") as f:
        json.dump(code_freq, f, indent=2)
    print(f"\n📊 Frequencies saved to: {freq_path}")

    # Save sorted solutions and source trace files
    sorted_codes = sorted(code_to_sources.items(), key=lambda x: len(x[1]), reverse=True)
    for i, (code, sources) in enumerate(sorted_codes, start=1):
        code_file = output_dir / f"{task_id}__#{i}.py"
        with open(code_file, "w") as f:
            f.write(code)

        # New: write source trace for this solution
        source_paths = [
            str(Path(output_base) / trial / filename)
            for (trial, filename) in sources
        ]
        sources_file = output_dir / f"{task_id}__#{i}_sources.json"
        with open(sources_file, "w") as f:
            json.dump(source_paths, f, indent=2)

        print(f"  💾 Saved solution #{i} → {code_file.name}")
        print(f"  📎 Source paths → {sources_file.name}")

    print(f"\n🎉 Done. Solutions written to: {output_dir}")



def main():
    parser = argparse.ArgumentParser(description="Analyze MBPP task convergence across trials.")
    parser.add_argument("root_dir", type=str, help="Root directory containing trial subdirectories.")
    parser.add_argument("task_id", type=int, help="The MBPP task_id to analyze.")
    parser.add_argument("--output-dir", type=str, default=".", help="Where to save output files.")
    args = parser.parse_args()

    code_to_sources, total_trials = analyze_task_across_trials(args.root_dir, args.task_id)
    if not code_to_sources:
        print(f"\n⚠️  No valid guided results found for task_id={args.task_id}")
    else:
        save_outputs(code_to_sources, total_trials, args.task_id, args.output_dir)


if __name__ == "__main__":
    main()
