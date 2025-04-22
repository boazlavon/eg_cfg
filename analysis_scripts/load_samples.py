import argparse
import os
import json
import re
from collections import defaultdict
from tqdm import tqdm


def find_and_analyze_task_id(root_dir, target_task_id):
    """
    Finds all JSON files with a specific task ID in a given root directory (recursively),
    loads their JSON content, extracts the 'code' property, and stores it along with
    identifiers (base_dir, gamma value).

    Args:
        root_dir (str): The root directory to search within.
        target_task_id (int): The specific task ID to look for in filenames.

    Returns:
        dict: A dictionary where keys are unique code strings and values are lists
              of identifiers (base_dir, gamma) for samples that produced that code.
    """
    code_to_identifiers = defaultdict(list)
    filepaths = []
    # print(f'Loading jsons with task_id={target_task_id}')
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".json") and f"task_id={target_task_id}" in filename:
                filepaths.append(os.path.join(dirpath, filename))

    # for filepath in tqdm(filepaths, desc=f"Processing task_id={target_task_id}"):
    for filepath in filepaths:
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
                code = data.get("code")
                if code is not None:
                    # Extract base directory name
                    base_dir = os.path.basename(
                        os.path.dirname(filepath)
                    )  # Get dir of the file

                    # Extract gamma value from filename
                    gamma_match = re.search(
                        r"_gamma=([^_]+)", os.path.basename(filepath)
                    )
                    gamma = None
                    if gamma_match:
                        try:
                            gamma_str = gamma_match.group(1)
                            if gamma_str.endswith(".json"):
                                gamma = float(gamma_str[:-5])
                            else:
                                gamma = float(gamma_str)
                        except ValueError:
                            pass
                            # print(f"Warning: Could not parse gamma value from filename: {os.path.basename(filepath)}")

                    identifier = (base_dir, gamma)
                    code_to_identifiers[code].append(identifier)
                else:
                    # print(f"Warning: 'code' property not found in {filepath}")
                    pass
        except Exception as e:
            pass
            # print(f"Error loading or processing {filepath}: {e}")

    return dict(code_to_identifiers)


JSON_FILE = "not_solved.json"
with open(JSON_FILE, "r") as f:
    tasks = json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Find and analyze JSON files for a specific task ID recursively and dump results to a task-specific JSON file with tqdm progress."
    )
    parser.add_argument(
        "root_dir", type=str, help="The root directory to search within."
    )
    parser.add_argument("task_id", type=int, help="The specific task ID to look for.")

    args = parser.parse_args()

    results = find_and_analyze_task_id(args.root_dir, args.task_id)
    output_filename = f"analysis/raw/code_analysis_{args.task_id}.json"
    indexed_results = {}
    indexed_output_filename = (
        f"analysis/indexed/indexed_code_analysis_{args.task_id}.jsonl"
    )
    with open(indexed_output_filename, "w") as outfile:
        for idx, result in enumerate(results.keys()):
            json.dump({"index": idx, "result": result}, outfile)
            outfile.write("\n")
            print(f"Result #{idx + 1}")
            print(result)
            print()

    with open(output_filename, "w") as outfile:
        json.dump(results, outfile, indent=4)

    for task in tasks:
        if task["task_id"] == args.task_id:
            break
    task_output_filename = (
        f"analysis/indexed/indexed_code_analysis_{args.task_id}t.json"
    )
    with open(task_output_filename, "w") as outfile:
        json.dump(task, outfile, indent=4)
    # print(f"Results dumped to {output_filename}")


if __name__ == "__main__":
    main()
