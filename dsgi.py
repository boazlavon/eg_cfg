import random

import pprint
import black
import torch
import argparse
import os
import json

from transformers import AutoTokenizer, AutoModelForCausalLM
from mbpp_utils import format_mbpp_prompt, read_problems, evaluate_solution
from code_generation import generate_solution, is_valid_python

MODEL_NAME = "meta-llama/Llama-3.2-1B"
# MODEL_NAME = "meta-llama/Llama-3.1-8B"
GAMMAS = (0.0, 0.1, 0.3, 0.5, 0.7, 0.8, 0.9)


def setup_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")
    return device


def load_model(model_name: str, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    return model, tokenizer


def generate_solutions(results_dir, start=0, end=None, gammas=GAMMAS):
    problems = read_problems()
    items = list(problems.items())

    if end is None:
        end = len(items)
    items = items[start:end]
    random.shuffle(items)

    device = setup_device()
    model, tokenizer = load_model(MODEL_NAME, device)
    solutions = {}

    for i, (_, problem) in enumerate(items):
        task_id = problem["task_id"]
        pprint.pprint(problem)
        print()
        print(f"task_id: {task_id}")
        for gamma in gammas:
            filename = f"task_id={task_id}_gamma={gamma}.json"
            filepath = os.path.join(results_dir, filename)
            if os.path.exists(filepath):
                print(f"solution exists: task_id={task_id}, gamma={gamma}")
                continue
            try:
                prompt, function_signature = format_mbpp_prompt(problem)
                new_tokens = generate_solution(
                    prompt,
                    function_signature,
                    problem["test_list"],
                    model,
                    tokenizer,
                    device,
                    gamma,
                )
                solution = f"{function_signature}\n{new_tokens}"

                assert is_valid_python(solution)
                solution = black.format_str(
                    solution, mode=black.FileMode(line_length=1024)
                )
                print(solution)
            except KeyboardInterrupt:
                exit(0)
            except:
                print("Invalid solution")
                continue
            print()

            test_cases = problem["test_list"]
            solution_results = {}
            for test_case in test_cases:
                try:
                    solution_results[test_case] = evaluate_solution(solution, test_case)
                except:
                    solution_results[test_case] = {
                        "result": False,
                        "time": -1,
                        "error": "Unknown",
                    }
                    print(f"Problem on executing test case {test_case}")
                    continue
            passed = all(
                [result_entry["result"] for result_entry in solution_results.values()]
            )
            correct = sum(
                [
                    int(result_entry["result"])
                    for result_entry in solution_results.values()
                ]
            )
            total = len(solution_results) * 1.0
            if not total:
                accuracy = 0
            else:
                accuracy = correct / total
            solution_entry = {
                "code": solution,
                "results": solution_results,
                "passed": passed,
                "accuracy": accuracy,
            }
            solutions[(task_id, gamma)] = solution_entry

            with open(filepath, "w") as f:
                json.dump(solutions[(task_id, gamma)], f, indent=2)

        # print('Problem')
        # print(problem['text'])
        # for gamma in gammas:
        #     print(f"LLM Solution: (gamma={gamma})")
        #     print(solutions[(task_id, gamma)])
        # print("Canonic Solution:")
        # print(problem['code'])
    return solutions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0, help="Start index of problems")
    parser.add_argument("--end", type=int, default=None, help="End index of problems")
    args = parser.parse_args()

    # Create results directory if it doesn't exist
    results_dir = os.path.join("results", "mbpp")
    os.makedirs(results_dir, exist_ok=True)

    generate_solutions(results_dir, start=args.start, end=args.end)


if __name__ == "__main__":
    main()
