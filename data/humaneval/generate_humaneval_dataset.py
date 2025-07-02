import ast
import time
import traceback
import json
import requests
import os
import re
from typing import List, Tuple
from datasets import load_dataset

from eg_cfg.eval_utils import run_tests
from eg_cfg.eg_cfg_session_manager import format_results


HUMANEVAL_JSON_PATH = "data/humaneval/humaneval.json"
HUMANEVAL_HF_PATH = "openai/openai_humaneval"
FW_URL = "https://api.fireworks.ai/inference/v1/chat/completions"
API_KEY = os.environ.get("FIREWORKS_API_KEY")
if not API_KEY:
    raise RuntimeError("FIREWORKS_API_KEY environment variable is not set.")

EXTRACT_TEST_CASES_PROMPT_TEMPLATE = """You are a code assistant. Extract all test cases from the following Python function docstring.

Respond with a valid JSON object **only** in the following format:
{{  
  "test_list": [
    ["function_call_1", "expected_output_1"],
    ["function_call_2", "expected_output_2"]
  ]
}}

Examples:

Input:
"prompt": "\\ndef circular_shift(x, shift):\\n    \\"\\"\\"Circular shift the digits of the integer x, shift the digits right by shift\\n    and return the result as a string.\\n    If shift > number of digits, return digits reversed.\\n    >>> circular_shift(12, 1)\\n    \\"21\\"\\n    >>> circular_shift(12, 2)\\n    \\"12\\"\\n    \\"\\"\\"\\n"

Output:
{{  
  "test_list": [
    ["circular_shift(12, 1)", "21"],
    ["circular_shift(12, 2)", "12"]
  ]
}}

Input:
"prompt": "\\n\\ndef fibfib(n: int):\\n    \\"\\"\\"FibFib sequence: fibfib(0) = 0, fibfib(1) = 0, fibfib(2) = 1.\\n    Then: fibfib(n) = fibfib(n-1) + fibfib(n-2) + fibfib(n-3)\\n    >>> fibfib(1)\\n    0\\n    >>> fibfib(5)\\n    4\\n    >>> fibfib(8)\\n    24\\n    \\"\\"\\"\\n"

Output:
{{  
  "test_list": [
    ["fibfib(1)", "0"],
    ["fibfib(5)", "4"],
    ["fibfib(8)", "24"]
  ]
}}

Input:
"prompt": "\\n\\ndef derivative(xs: list):\\n    \\"\\"\\" xs represent coefficients of a polynomial.\\n    xs[0] + xs[1] * x + xs[2] * x^2 + ....\\n    Return derivative of this polynomial in the same form.\\n    >>> derivative([3, 1, 2, 4, 5])\\n    [1, 4, 12, 20]\\n    >>> derivative([1, 2, 3])\\n    [2, 6]\\n    \\"\\"\\"\\n"

Output:
{{  
  "test_list": [
    ["derivative([3, 1, 2, 4, 5])", "[1, 4, 12, 20]"],
    ["derivative([1, 2, 3])", "[2, 6]"]
  ]
}}

Now extract test cases from this prompt:

Input:
"prompt": "{prompt}"
"""

GENERATE_TEST_CASES_FEW_SHOT_PROMPT = """You are a Python code assistant.

Given a function **prompt string** that includes the function signature and docstring (but no examples), your task is to generate exactly 3 representative test cases per top-level function.

Respond with a valid JSON object in this format:
{{  
  "test_list": [
    ["function_call(arguments)", "expected_output"],
    ...
  ]
}}

Each test case must be on its own line inside the list. Use only data types and behaviors consistent with the function's purpose.

Examples:

Input:
"prompt": "\\ndef square(x):\\n    \\\"\\\"\\\"Returns the square of a number.\\\"\\\"\\\"\\n    return x * x\\n"

Output:
{{  
  "test_list": [
    ["square(2)", "4"],
    ["square(5)", "25"],
    ["square(0)", "0"]
  ]
}}

Input:
"prompt": "\\ndef is_even(n):\\n    \\\"\\\"\\\"Returns True if the number is even, else False.\\\"\\\"\\\"\\n    return n % 2 == 0\\n"

Output:
{{  
  "test_list": [
    ["is_even(2)", "True"],
    ["is_even(3)", "False"],
    ["is_even(0)", "True"]
  ]
}}

Now generate test cases for this prompt:

Input:
"prompt": "{prompt}"

Output:
"""


def extract_test_list_from_prompt(prompt: str, template) -> list:
    api_key = API_KEY
    url = FW_URL

    full_prompt = template.format(prompt=prompt.strip())

    payload = {
        "model": "accounts/fireworks/models/deepseek-v3-0324",
        "max_tokens": 2048,
        "top_p": 1,
        "top_k": 40,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "temperature": 0.6,
        "messages": [{"role": "user", "content": full_prompt}],
    }

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    response = requests.post(url, headers=headers, data=json.dumps(payload))
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]

    try:
        # Try parsing the entire message
        parsed = json.loads(content)
        content = parsed["test_list"]
    except json.JSONDecodeError:
        # Attempt to extract the JSON block manually
        match = re.search(r'\{\s*"test_list"\s*:\s*(\[[\s\S]+?\])\s*\}', content)
        if not match:
            raise ValueError("Could not extract 'test_list' from model response.")

        test_list_str = match.group(1)
        test_list = json.loads(test_list_str)
        content = test_list
    return content


def extract_signature_by_line(code: str, entry_point: str) -> str:
    """
    Finds the first line in the code that starts with 'def {entry_point}' and returns it without the trailing colon.

    Args:
        code (str): Full Python source code.
        entry_point (str): Function name to search for.

    Returns:
        str: Function signature line without the colon.
    """
    for line in code.splitlines():
        line = line.strip()
        if line.startswith(f"def {entry_point}"):
            return line.rstrip(":").strip()
    raise ValueError(f"Function '{entry_point}' not found.")


def extract_instruction_and_tests_clean(
    prompt_code: str, entry_point: str
) -> Tuple[str, List[Tuple[str, str]], str]:
    """
    Extracts instruction and test cases from prompt docstring.
    Ensures:
      - Instruction contains no '>>>'
      - Test case inputs/results contain no '>>>'

    Returns:
        - instruction: str
        - test_cases: List of (invocation, result)
        - cleaned_prompt: str (prompt with instruction removed from docstring)
    """
    # Parse the function
    tree = ast.parse(prompt_code)
    func_node = next(
        (
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == entry_point
        ),
        None,
    )
    if not func_node:
        raise ValueError(f"Function '{entry_point}' not found.")

    # Extract full (raw) docstring
    docstring = ast.get_docstring(func_node, clean=False)
    if not docstring:
        return "", [], prompt_code

    # 1. Extract instruction (everything before first >>>)
    split_doc = docstring.split(">>>", maxsplit=1)
    instruction = split_doc[0].strip()
    assert ">>>" not in instruction, "Instruction must not contain '>>>'"

    remaining = split_doc[1] if len(split_doc) > 1 else ""

    # 2. Extract test cases from the remaining docstring
    test_cases = []
    for segment in remaining.split(">>>"):
        lines = segment.strip().splitlines()
        if not lines:
            continue
        invocation = lines[0].strip()
        result = lines[1].strip() if len(lines) > 1 else ""

        assert ">>>" not in invocation, "Invocation must not contain '>>>'"
        assert ">>>" not in result, "Result must not contain '>>>'"

        test_cases.append((invocation, result))

    # 3. Clean the prompt by removing the instruction from the docstring
    new_docstring = '"""\n' + "\n>>>".join([""] + remaining.split(">>>")) + '\n"""'
    updated_prompt = prompt_code.replace(docstring, new_docstring)

    return instruction, test_cases, updated_prompt


def test_case_to_assert(invocation: str, expected: str) -> str:
    """Convert a (input, expected_output) pair to a Python assert statement."""
    return f"assert {invocation} == {expected}"


def validate_sample(example):
    test_cases = [
        test_case_to_assert(invocation, expected)
        for (invocation, expected) in example["test_list"]
    ]
    solution = example["prompt"] + example["canonical_solution"]
    solution_results = run_tests(solution, test_cases)
    solution_entry = format_results(solution, solution_results, None, None)
    return solution_entry["passed"]


def extract_determenistic_pattern_dataset(humaneval):
    extracted_components = {
        example["task_id"]: extract_instruction_and_tests_clean(
            example["prompt"], example["entry_point"]
        )
        for example in humaneval["test"]
        if len(
            extract_instruction_and_tests_clean(
                example["prompt"], example["entry_point"]
            )[1]
        )
        == example["prompt"].count(">>>")
    }
    assert len(extracted_components) == len(humaneval["test"])
    dataset = {}
    for example in humaneval["test"]:
        task_id = example["task_id"]
        # print(task_id)
        dataset[task_id] = example
        instruction, test_cases, _ = extracted_components[task_id]
        function_sig = extract_signature_by_line(
            example["prompt"], example["entry_point"]
        )
        solution = example["prompt"] + example["canonical_solution"]
        instruction = instruction.replace("\n", "")
        instruction = instruction.replace("    ", " ")
        instruction = instruction.replace("   ", " ")
        dataset[task_id]["text"] = instruction
        dataset[task_id]["function_signature"] = function_sig
        assert function_sig in example["prompt"]
        dataset[task_id]["test_list"] = test_cases
        dataset[task_id]["code"] = solution
    return dataset


def extract_testcases_with_llm(dataset):
    llm_based_dataset = {}
    RETRIES = 3
    for task_id, example in dataset.items():
        if example["test_list"] and not validate_sample(example):
            test_list = example["test_list"]
            print(f"Invalid test cases: {test_list}")
            example["test_list"] = []

        if not example["test_list"] or any(
            example[0] and not example[1] for example in example["test_list"]
        ):
            print(task_id)
            for i in range(2 * RETRIES):
                try:
                    if i < RETRIES:
                        test_list = extract_test_list_from_prompt(
                            example["prompt"], EXTRACT_TEST_CASES_PROMPT_TEMPLATE
                        )
                    else:
                        test_list = extract_test_list_from_prompt(
                            example["prompt"], GENERATE_TEST_CASES_FEW_SHOT_PROMPT
                        )
                    assert validate_sample(example), f"Invalid test cases: {test_list}"
                except Exception as e:
                    test_list = []
                    print(traceback.format_exc())
                    time.sleep(5)
                    continue
                example["test_list"] = test_list
                break
        llm_based_dataset[task_id] = example
    return llm_based_dataset


def build_dataset_with_tests_cases():
    humaneval = load_dataset(HUMANEVAL_HF_PATH)
    dataset = extract_determenistic_pattern_dataset(humaneval)
    llm_based_dataset = extract_testcases_with_llm(dataset)
    return llm_based_dataset


def validate_tests(dataset):
    solutions = {}
    unsolved = []
    unsolved = set(unsolved)
    for task_id, example in dataset.items():
        task_id = example["task_id"]
        test_cases = [
            test_case_to_assert(invocation, expected)
            for (invocation, expected) in example["test_list"]
        ]
        if not test_cases:
            print(f"problem! {task_id}")
            entry = {
                "code": solution,
                "results": [],
                "passed": False,
                "accuracy": 0,
                "general_error": None,
                "has_testcase_error": None,
            }
            solutions[task_id] = entry
            continue
        solution = example["prompt"] + example["canonical_solution"]
        solution_results = run_tests(solution, test_cases)
        general_error = None
        tb = None
        solution_entry = format_results(solution, solution_results, general_error, tb)
        solutions[task_id] = solution_entry
        print(task_id)

    invalid_entries = {
        task_id
        for task_id, solution_entry in solutions.items()
        if not solution_entry["passed"]
    }
    print()
    print(f"Total: {len(solutions)}")
    print(f"Valid Entries: {len(solutions) - len(invalid_entries)}")

    # HumanEval116 canoncial solution is wrong, so its an invalid entry
    # HumanEval47 has a wrong test case

    print(f"Invalid Entries: {len(invalid_entries)}")
    invalid_entries = list(invalid_entries)
    invalid_entries.sort()
    print(invalid_entries)
    return solutions, invalid_entries


def main():
    dataset = None
    try:
        with open(HUMANEVAL_JSON_PATH, "r") as f:
            dataset = json.load(f)
    except:
        pass

    if not dataset:
        dataset = build_dataset_with_tests_cases()
        with open(HUMANEVAL_JSON_PATH, "w") as f:
            json.dump(dataset, f, indent=2)

    validate_tests(dataset)


if __name__ == "__main__":
    main()
