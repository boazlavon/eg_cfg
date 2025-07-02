import json


def extract_solution_and_tests(json_path, task_id):
    with open(json_path, "r") as f:
        data = json.load(f)

    task = data.get(task_id)
    if task is None:
        print(f"❌ Task ID '{task_id}' not found.")
        return

    solution = task["prompt"] + task["canonical_solution"]
    print("=== Prompt ===")
    print(solution)
    test = task["test"]
    print("=== RawTest ===")
    print(test)
    exec(solution, globals())
    exec(task["test"], globals())


# Example usage:
import sys

task_id = sys.argv[1]
extract_solution_and_tests(
    "/a/home/cc/students/cs/boazlavon/code/web/clean4/eg_cfg/data/humaneval/humaneval.json",
    f"HumanEval/{task_id}",
)
