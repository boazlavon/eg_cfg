import random
import pprint
import black
import argparse
import os
import json

from mbpp_utils import (
    format_mbpp_prompt,
    load_mbpp_problems,
    evaluate_solution,
    run_tests,
)
from code_generation_utils import generate_code_solutions, is_valid_python
from dsgi_manager import DsgiManager, TASK__CODE_GENERATION
from model_utils import setup_device, load_model
from code_generation_adapter import (
    DYNAMIC_SIGNAL__PARTIAL_EXECUTION,
    DYNAMIC_SIGNAL__NEAREST_FUTURE_EXECUTION,
)

MODEL_NAME = "meta-llama/Llama-3.2-1B"
# MODEL_NAME = "google/gemma-3-1b-it"
# MODEL_NAME = "meta-llama/Llama-3.1-8B"
NO_GUIDANCE_SIMPLE_PROMPT_GAMMA = -0.1
GAMMAS = (NO_GUIDANCE_SIMPLE_PROMPT_GAMMA, 0.0, 0.1, 0.3, 0.5, 0.7, 0.8, 0.9)
FILENAME_TEMPLATE = "task_id={task_id}_gamma={gamma}.json"


def should_skip(results_dir, task_id, gamma):
    filepath = get_solution_filepath(results_dir, task_id, gamma)
    if os.path.exists(filepath):
        print(f"🟡 Solution exists: task_id={task_id}, gamma={gamma}")
        return True
    # touch the file to reserve it
    with open(filepath, "a"):
        pass
    return False


def try_generate_code_solutions(model, tokenizer, device, problem, gamma):
    simple_prompt = gamma == NO_GUIDANCE_SIMPLE_PROMPT_GAMMA
    prompt, function_signature = format_mbpp_prompt(problem, simple_prompt)
    test_cases = problem["test_list"]

    dsgi_manager = None
    if not simple_prompt:
        task_kwargs = {
            "model": model,
            "tokenizer": tokenizer,
            "device": device,
            "function_signature": function_signature,
            "test_cases": test_cases,
            "initial_prompt": prompt,
            "dynamic_signals": (
                DYNAMIC_SIGNAL__PARTIAL_EXECUTION,
                DYNAMIC_SIGNAL__NEAREST_FUTURE_EXECUTION,
            ),
            "nearest_future_samples": 5,
            "nearest_future_lines": 3,
        }
        task = TASK__CODE_GENERATION
        dsgi_manager = DsgiManager(tokenizer, task, task_kwargs, gamma)

    function_code = generate_code_solutions(
        model, tokenizer, device, prompt, dsgi_manager
    )
    solution = f"{function_signature}\n{function_code}"

    assert is_valid_python(solution)
    solution = black.format_str(solution, mode=black.FileMode(line_length=1024))
    return solution


def format_results(solution, results, general_error):
    passed = all(r["result"] for r in results.values())
    correct = sum(int(r["result"]) for r in results.values())
    total = len(results)
    accuracy = correct / total if total else 0.0
    has_testcase_error = all([bool(result["error"]) for result in results.values()])
    return {
        "code": solution,
        "results": results,
        "passed": passed,
        "accuracy": accuracy,
        "general_error": general_error,
        "has_testcase_error": has_testcase_error,
    }


def get_solution_filepath(results_dir, task_id, gamma):
    filename = FILENAME_TEMPLATE.format(task_id=task_id, gamma=gamma)
    return os.path.join(results_dir, filename)


def generate_mbpp_solutions(results_dir, start=0, end=None, gammas=GAMMAS):
    device = setup_device()
    model, tokenizer = load_model(MODEL_NAME, device)
    solutions = {}

    problems = load_mbpp_problems()
    problems = list(problems.items())

    if end is None:
        end = len(problems)
    problems = problems[start:end]
    random.shuffle(problems)

    for _, problem in problems:
        problem_solved = False
        task_id = problem["task_id"]
        test_cases = problem["test_list"]

        print(f"task_id: {task_id}")
        pprint.pprint(problem)
        print()

        for gamma in gammas:
            if gamma > 0 and problem_solved:
                print(f"Skip gamma={gamma} problem is solved")
                continue

            general_error = None
            if should_skip(results_dir, task_id, gamma):
                continue

            try:
                solution = try_generate_code_solutions(
                    model, tokenizer, device, problem, gamma
                )
                print(solution)
            except KeyboardInterrupt:
                exit(1)
            except AssertionError as e:
                solution = None
                general_error = str(type(e))
                raise e
            except Exception as e:
                solution = None
                general_error = str(type(e))
                raise e

            print()
            solution_results = run_tests(solution, test_cases)
            solution_entry = format_results(solution, solution_results, general_error)
            filepath = get_solution_filepath(results_dir, task_id, gamma)
            with open(filepath, "w") as f:
                json.dump(solution_entry, f, indent=2)
            solutions[(task_id, gamma)] = solution_entry
            if solution_entry["passed"]:
                problem_solved = True

    return solutions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0, help="Start index of problems")
    parser.add_argument("--end", type=int, default=None, help="End index of problems")
    parser.add_argument(
        "--model-name", type=str, default=MODEL_NAME, help="Name of the model used"
    )
    args = parser.parse_args()

    # Create results directory if it doesn't exist
    sanitized_model_name = args.model_name.replace("/", "_")
    results_dir = os.path.join("results", "mbpp", sanitized_model_name)
    os.makedirs(results_dir, exist_ok=True)

    generate_mbpp_solutions(results_dir, start=args.start, end=args.end)


if __name__ == "__main__":
    main()
