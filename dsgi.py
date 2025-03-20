import random

random.seed(43)
import ast
import re

import black
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from eval_mbpp import evaluate_mbpp_from_dict

MODEL_NAME = "meta-llama/Llama-3.2-1B"

INSTRUCTION_TEXT = """\
### Instruction:
{problem_text}

Write a Python function that satisfies the following test cases:
{test_cases}

Your solution must follow this function signature:
{function_signature}

Your solution should be written in **as many lines as possible**.
This ensures that **prefixes of your function remain valid Python programs**, 
allowing incremental execution and debugging.

Write the function **step by step**, progressively introducing variables and logic.
For example:
- First, define the function.
- Then, initialize variables.
- Finally, implement logic in separate steps.

**Avoid using list comprehensions, lambda functions, or overly compact one-liners.**
Instead, follow these guidelines:**

Avoid list comprehensions, use loops instead:
Incorrect:
def square_numbers(lst):
    return [x ** 2 for x in lst]  # One-liner comprehension

Correct:
def square_numbers(lst):
    squares = []  # Use a variable to store results
    for num in lst:
        squared_value = num ** 2  # Assign intermediate results
        squares.append(squared_value)
    return squares

Avoid inline expressions, use variables instead
Incorrect:
def calculate_area(length, width):
    return (length * width) / 2  # Inline expression

Correct:
def calculate_area(length, width):
    product = length * width  # Store intermediate result
    area = product / 2
    return area

Incorrect:
result.append(x + y)

Correct:
z = x + y
result.append(z)

Incorrect:
def compute_value(a, b, c):
    return (a + b) * (c / (a - b) + (a * c) / (b + c))  # ❌ Too complex to read

Correct:
def compute_value(a, b, c):
    term1 = a + b  # ✅ Compute first term separately
    term2 = a - b  # ✅ Store denominator separately
    term3 = c / term2  # ✅ Compute first fraction
    term4 = a * c / (b + c)  # ✅ Compute second fraction
    result = term1 * (term3 + term4)  # ✅ Combine step by step
    return result

### Response:
{function_signature}

"""


def make_executable_partial_code(prompt, new_tokens):
    # Randomly cut the generated code at some point
    for cut_index in range(len(new_tokens)):
        cut_code = f"{prompt}{new_tokens[:cut_index]}"
        executable_code = make_executable(prompt, cut_code)
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print(f"Cut Solution #{cut_index}:\n{new_tokens[:cut_index]}")
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print(f"Executable Solution #{cut_index}:\n{executable_code}")
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print()


def make_executable(prompt: str, code: str, fallback_to_prompt: bool = True) -> str:
    lines = code.split("\n")
    fixed_code = ""

    while lines:
        fixed_code = "\n".join(lines)
        if is_valid_python(fixed_code) and fixed_code.startswith(prompt):
            break

        # Remove last line and try again
        last_line = lines.pop()
        if not lines:
            break  # Stop if there are no lines left

        fixed_code = "\n".join(lines)
        if is_valid_python(fixed_code) and fixed_code.startswith(prompt):
            break

        # If removing doesn't work, replace last line with 'pass' (preserving indentation)
        indent = re.match(r"\s*", last_line).group(0)
        lines.append(f"{indent}pass")
        fixed_code = "\n".join(lines)
        if is_valid_python(fixed_code) and fixed_code.startswith(prompt):
            break
        lines.pop()  # Remove the pass if it's still invalid

    if (
        not is_valid_python(fixed_code) or not fixed_code.startswith(prompt)
    ) and fallback_to_prompt:
        prompt_lines = prompt.split("\n")
        last_line = prompt_lines[-1]
        indent = re.match(r"\s*", last_line).group(0)
        fixed_code = f"{prompt}"
        fixed_code = black.format_str(fixed_code, mode=black.FileMode())

    return fixed_code


def is_valid_python(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


class CodeGenStopCriteria(torch.utils.data.Dataset):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.generated_text = ""
        self.previous_newline_index = -1
        self.remove_last_newline = False

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        new_token = self.tokenizer.decode(input_ids[0][-1])
        if "<|end_of_text|>" in new_token:
            new_token = new_token.replace("<|end_of_text|>", "")
            self.generated_text += new_token
            return True

        self.generated_text += new_token
        if "\n" in new_token:
            if self.previous_newline_index != -1:
                current_newline_index = len(self.generated_text) - 1
                substring = self.generated_text[
                    self.previous_newline_index + 1 : current_newline_index
                ]

                if substring.startswith(" ") or substring.startswith("\t"):
                    self.previous_newline_index = current_newline_index
                    return False
                else:
                    # print(self.previous_newline_index, current_newline_index, substring)
                    self.remove_last_newline = True
                    self.generated_text = self.generated_text[
                        : self.previous_newline_index
                    ]
                    self.generated_text = self.generated_text.replace("\n\n", "\n")
                    return True
            else:
                self.previous_newline_index = len(self.generated_text) - 1
            return False

        return False


def generate_solution(prompt, model, tokenizer, device):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    stop_criteria = CodeGenStopCriteria(tokenizer)

    with torch.no_grad():
        _ = model.generate(
            **inputs,
            max_new_tokens=512,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            stopping_criteria=[stop_criteria],
        )

    # _ = tokenizer.decode(outputs[0], skip_special_tokens=True)
    new_tokens = stop_criteria.generated_text
    return new_tokens


def extract_function_signature(code):
    match = re.search(r"def\s+\w+\s*\(.*\):", code)
    return match.group(0) if match else None


def format_mbpp_prompt(problem):
    function_signature = extract_function_signature(problem["code"])
    assert function_signature, "Function signature could not be extracted."

    # Extract and format test cases using Black
    formatted_test_cases = ""
    for test in problem["test_list"]:
        formatted_test_cases += black.format_str(
            test, mode=black.FileMode(line_length=1024)
        )

    prompt = INSTRUCTION_TEXT.format(
        problem_text=problem["text"], function_signature=function_signature,
        test_cases=formatted_test_cases
    )

    return prompt, function_signature


def setup_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")
    return device


def load_model(model_name: str, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    return model, tokenizer


def read_problems():
    dataset = load_dataset("mbpp")
    problems = {str(i): example for i, example in enumerate(dataset["test"])}
    return problems


def generate_solutions(k=20):
    problems = read_problems()
    results = {}
    items = list(problems.items())
    random.shuffle(items)

    device = setup_device()
    model, tokenizer = load_model(MODEL_NAME, device)
    solutions = {}

    for i, (_, problem) in enumerate(items):
        # if i >= k:
        #     break
        task_id = problem['task_id']
        print(f"\nTask ID: {task_id}")
        try:
            prompt, function_signature = format_mbpp_prompt(problem)
            # print(function_signature)
            # print(prompt)
            new_tokens = generate_solution(prompt, model, tokenizer, device)
            solution = f"{function_signature}\n{new_tokens}"

            assert is_valid_python(solution)
            solution = black.format_str(solution, mode=black.FileMode(line_length=1024))
            print(solution)
        except:
            print("Invalid solution")
            continue
        print()
        solutions[task_id] = solution

        # If needed, you can extract test cases from MBPP's "test_list" field
        # test_cases = problem["test_list"]
    return solutions

def main():
    solutions = generate_solutions()
    results = evaluate_mbpp_from_dict(solutions)
    print(results)

if __name__ == "__main__":
    main()
