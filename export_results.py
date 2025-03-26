import os
import json
import csv
import re
import argparse
from datetime import datetime
from collections import defaultdict
from dsgi import NO_GUIDANCE_SIMPLE_PROMPT_GAMMA

NO_GUIDANCE_GAMMAS = [NO_GUIDANCE_SIMPLE_PROMPT_GAMMA, 0.0]


def extract_data_from_filenames(directory):
    """
    Extracts task_id, gamma, accuracy, and passed data from JSON files in a directory.

    Args:
        directory (str): The path to the directory containing the JSON files.

    Returns:
        tuple: A tuple containing two dictionaries, one for accuracy and one for passed data.
               Each dictionary is structured as {task_id: {gamma: value}}.
    """
    accuracy_data = {}
    passed_data = {}

    for filename in os.listdir(directory):
        if (
            filename.endswith(".json")
            and "task_id=" in filename
            and "gamma=" in filename
        ):
            match = re.search(r"task_id=(\d+)_gamma=(-?\d*\.?\d+)", filename)
            if match:
                task_id = int(match.group(1))
                gamma = float(match.group(2))

                filepath = os.path.join(directory, filename)
                try:
                    with open(filepath, "r") as f:
                        data = json.load(f)
                        accuracy = data.get("accuracy")
                        passed = data.get("passed")

                        if task_id not in accuracy_data:
                            accuracy_data[task_id] = {}
                        if task_id not in passed_data:
                            passed_data[task_id] = {}

                        if accuracy is not None:
                            accuracy_data[task_id][gamma] = accuracy
                        if passed is not None:
                            passed_data[task_id][gamma] = passed
                except (FileNotFoundError, json.JSONDecodeError) as e:
                    print(f"Error processing {filename}: {e}")

    return accuracy_data, passed_data


def write_data_to_csv(data, filepath, value_type):
    """
    Writes the given data (task_id/gamma) to a CSV file.

    Args:
        data (dict): The data dictionary to write.
        filepath (str): The path to the CSV file.
        value_type (str): The type of value being written (e.g., "accuracy", "passed").
    """
    with open(filepath, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        # Get unique gammas and sort them
        gammas = sorted(
            list(
                set(gamma for task_data in data.values() for gamma in task_data.keys())
            )
        )
        header = ["task_id"] + [str(gamma) for gamma in gammas]
        writer.writerow(header)

        for task_id, task_data in sorted(data.items()):
            row = [task_id] + [task_data.get(gamma, "") for gamma in gammas]
            writer.writerow(row)


# def filter_and_write_passed_csv(passed_data, filepath):
#     """
#     Filters and writes the passed data to a CSV file, showing only cases where gamma=0 has influence.
#     """
#     filtered_data = {}
#     for task_id, task_data in passed_data.items():
#         if 0.0 in task_data:
#             zero_gamma_value = task_data[0.0]
#             for gamma, value in task_data.items():
#                 if gamma != 0.0 and value != zero_gamma_value:
#                     filtered_data[task_id] = task_data
#                     break
#     return filtered_data


def filter_and_write_passed_csv(passed_data, gammas_to_check):
    filtered_data = {}

    for task_id, task_data in passed_data.items():
        # Collect OR from only gammas that exist in the data
        or_values = [
            task_data[gamma] for gamma in gammas_to_check if gamma in task_data
        ]
        if not or_values:
            continue  # skip if no relevant gammas found
        or_result = any(or_values)

        # Check if there’s any other gamma with a different result than the OR
        has_difference = any(value != or_result for gamma, value in task_data.items())
        if has_difference:
            new_task_data = dict(task_data)  # shallow copy
            new_task_data[-1] = or_result  # inject new synthetic gamma
            filtered_data[task_id] = new_task_data

    return filtered_data


def count_gamma_values(data, gammas, is_passed=False):
    result = defaultdict(lambda: defaultdict(int))
    for task_id, task_data in data.items():
        for gamma in gammas:
            if gamma in task_data:
                result[gamma][task_data[gamma]] += 1
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Extract and process data from JSON files."
    )
    parser.add_argument(
        "directory",
        default="results/mbpp",
        nargs="?",
        help="Directory containing JSON files.",
    )
    args = parser.parse_args()

    directory = args.directory

    if not os.path.exists(directory):
        print(f"Directory '{directory}' not found.")
        return

    accuracy_data, passed_data = extract_data_from_filenames(directory)

    csv_dir = os.path.join(directory, "csvs")
    os.makedirs(csv_dir, exist_ok=True)  # creates the directory if it does not exist.

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    accuracy_filepath = os.path.join(csv_dir, f"accuracy_{timestamp}.csv")
    passed_filepath = os.path.join(csv_dir, f"passed_{timestamp}.csv")

    write_data_to_csv(accuracy_data, accuracy_filepath, "accuracy")
    filtered_passed_data = filter_and_write_passed_csv(passed_data, NO_GUIDANCE_GAMMAS)
    write_data_to_csv(filtered_passed_data, passed_filepath, "passed")

    NO_GUIDANCE_GAMMAS.append(-1)
    accuracy_counts = count_gamma_values(accuracy_data, NO_GUIDANCE_GAMMAS)
    passed_counts = count_gamma_values(
        filtered_passed_data, NO_GUIDANCE_GAMMAS, is_passed=True
    )

    print(f"\nCSV files were created in '{csv_dir}'.")
    print(accuracy_filepath)
    print(passed_filepath)

    print("\nAccuracy Value Counts by Gamma:")
    for gamma in sorted(accuracy_counts.keys()):
        print(f"Gamma = {gamma}:")
        for value, count in sorted(accuracy_counts[gamma].items()):
            print(f"  {value}: {count}")

    print("\nPassed Value Counts by Gamma:")
    for gamma in sorted(passed_counts.keys()):
        print(f"Gamma = {gamma}:")
        print(f"Total improvments: {passed_counts[gamma][False]}")


if __name__ == "__main__":
    main()
