import random

random.seed(43)

import black
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from mbpp_utils import evaluate_mbpp_from_dict, format_mbpp_prompt, read_problems
from extract_executable import make_executable, is_valid_python
from code_generation import generate_solution


def setup_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")
    return device


def load_model(model_name: str, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    return model, tokenizer


MODEL_NAME = "meta-llama/Llama-3.2-1B"
# MODEL_NAME = "meta-llama/Llama-3.1-8B"


def generate_solutions(k=2):
    problems = read_problems()
    items = list(problems.items())
    random.shuffle(items)

    device = setup_device()
    model, tokenizer = load_model(MODEL_NAME, device)
    solutions = {}

    for i, (_, problem) in enumerate(items):
        if i >= k:
            break
        task_id = problem["task_id"]
        print(f"\nTask ID: {task_id}")
        try:
            prompt, function_signature = format_mbpp_prompt(problem)
            # print(function_signature)
            # print(prompt)
            new_tokens = generate_solution(
                prompt, function_signature, model, tokenizer, device
            )
            solution = f"{function_signature}\n{new_tokens}"

            assert is_valid_python(solution)
            solution = black.format_str(solution, mode=black.FileMode(line_length=1024))
            print(solution)
        except KeyboardInterrupt:
            exit(0)
        except:
            print("Invalid solution")
            continue
        print()
        solutions[task_id] = solution
    return solutions


def main():
    solutions = generate_solutions()
    results = evaluate_mbpp_from_dict(solutions)
    print(results)


if __name__ == "__main__":
    main()
