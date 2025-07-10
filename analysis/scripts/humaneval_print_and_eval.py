import argparse
import json


def extract_solution_and_tests(json_path, task_id):
    with open(json_path, "r") as f:
        data = json.load(f)

    task = data.get(task_id)
    if task is None:
        print(f"[ERROR] Task ID '{task_id}' not found in '{json_path}'.")
        return

    solution_code = task["prompt"] + task["canonical_solution"]
    test_code = task["test"]

    print("=== Prompt and Solution ===")
    print(solution_code)
    print("\n=== Raw Test ===")
    print(test_code)

    print("\n=== Executing Solution and Test ===")
    exec(solution_code, globals())
    exec(test_code, globals())


def main():
    parser = argparse.ArgumentParser(
        description="Extract and execute a HumanEval solution and its test case."
    )
    parser.add_argument("task_id", help="HumanEval Task ID (HumanEval/TASK_ID)")
    parser.add_argument("json_path", help="Path to the HumanEval JSON file")

    args = parser.parse_args()
    extract_solution_and_tests(args.json_path, args.task_id)


if __name__ == "__main__":
    main()
