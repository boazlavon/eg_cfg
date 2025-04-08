import re
import json
import subprocess
import tempfile
import time
import black
import traceback
from datasets import load_dataset
from consts import *

CUSTOM_INSTRUCTION_TEXT = """\
### Instruction:
{problem_text}

Write a Python function that satisfies the following test cases:
{test_cases}

Your solution should be written in as many lines as possible.
This ensures that prefixes of your function remain valid Python programs.
Allowing **incremental execution and debugging**.

Write the function **step by step**, progressively introducing variables and logic.
Avoid using list comprehensions, lambda functions, or overly compact one-liners.
Instead, follow these guidelines:**

Avoid list comprehensions, use loops instead:
Incorrect:
def square_numbers(lst):
    return [x ** 2 for x in lst]

Correct:
def square_numbers(lst):
    squares = []
    for num in lst:
        squared_value = num ** 2
        squares.append(squared_value)
    return squares

Avoid inline expressions, use variables instead
Incorrect:
def calculate_area(length, width):
    return (length * width) / 2

Correct:
def calculate_area(length, width):
    product = length * width
    area = product / 2
    return area

Incorrect:
result.append(x + y)

Correct:
z = x + y
result.append(z)

Incorrect:
def compute_value(a, b, c):
    return (a + b) * (c / (a - b) + (a * c) / (b + c))

Correct:
def compute_value(a, b, c):
    term1 = a + b 
    term2 = a - b 
    term3 = c / term2 
    term4 = a * c / (b + c)
    result = term1 * (term3 + term4)
    return result

### Response:
{function_signature}
"""


TASK_HEADER = "### Task"
GOAL_INSTRUCTION = (
    "### Your goal is to write a Python function that solves the problem above."
)
EXAMPLES_HEADER = "### Here are some examples:"

PROMPT_TEMPLATE = """{task_header}
{text}

{goal_instruction}
{examples_block}

{function_signature}
"""


def run_tests(solution, test_cases):
    results = {}
    for test_case in test_cases:
        if solution is None:
            results[test_case] = {
                "result": False,
                "time": -1,
                "error": "GenerationError",
            }
            continue

        try:
            results[test_case] = evaluate_solution(solution, test_case)
        except Exception as e:
            tb = traceback.format_exc()
            results[test_case] = {
                "result": False,
                "time": -1,
                "error": str(type(e)),
                "tb": tb,
            }
            print(f"Problem executing test case: {test_case}")
    return results


def evaluate_solution(code, test_case, timeout=10):
    test_passed = False
    error = None
    test_code = f"{code}\n{test_case}"
    test_code = black.format_str(test_code, mode=black.FileMode(line_length=1024))

    with tempfile.NamedTemporaryFile(suffix=".py", mode="w") as temp_file:
        temp_file.write(test_code)
        temp_file.flush()

        start_time = time.time()
        try:
            result = subprocess.run(
                ["python", temp_file.name],
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            if result.returncode == 0 and not "Traceback" in result.stderr:
                test_passed = True

        except subprocess.TimeoutExpired:
            error = "Timeout"
            pass
        except Exception as e:
            error = "Exception"
            pass
        finally:
            end_time = time.time()
            delta_time = end_time - start_time

    result_entry = {"result": test_passed, "time": delta_time, "error": error}
    return result_entry


def extract_function_signature(code):
    # match = re.search(r"def\s+\w+\s*\(.*\):", code)
    match = re.search(r"def\s+\w+\s*\(.*\)\s*:", code)
    return match.group(0) if match else None


def parse_mbpp_assert_statement(assert_statement):
    ASSERT_PATTERN = r"^assert\s*\(*\s*(\w+)\s*\((.*)\)\s*\)*\s*==\s*(.+)$"
    match = re.match(ASSERT_PATTERN, assert_statement.strip())
    if not match:
        print(assert_statement)
        raise ValueError(f"Invalid assert statement format. {assert_statement}")

    function_name = match.group(1)
    args_str = match.group(2)
    args_str = f"({args_str})"
    expected_result_str = match.group(3)
    return (function_name, args_str, expected_result_str)


def format_simple_mbpp_prompt(problem, function_signature):
    task_description = problem["text"].strip()
    test_list = problem["test_list"]

    examples = []
    function_name = None

    for test in test_list:
        try:
            fn_name, args_str, expected = parse_mbpp_assert_statement(test)
            function_name = function_name or fn_name
            examples.append(
                f"# Example {len(examples)+1}:\n# Input: {function_name}{args_str}\n# Output: {expected}"
            )
        except ValueError:
            continue

    examples_body = "\n".join(examples) if examples else ""
    examples_block = f"\n{EXAMPLES_HEADER}\n{examples_body}" if examples_body else ""

    return PROMPT_TEMPLATE.format(
        task_header=TASK_HEADER,
        text=task_description,
        goal_instruction=GOAL_INSTRUCTION,
        examples_block=examples_block,
        function_signature=function_signature,
    ).strip()


def format_custom_mbpp_prompt(problem, function_signature):
    formatted_test_cases = ""
    for test in problem["test_list"]:
        formatted_test_cases += black.format_str(
            test, mode=black.FileMode(line_length=1024)
        )

    prompt = CUSTOM_INSTRUCTION_TEXT.format(
        problem_text=problem["text"],
        function_signature=function_signature,
        test_cases=formatted_test_cases,
    )

    return prompt


def format_mbpp_prompt(problem, simple_prompt=False):
    function_signature = extract_function_signature(problem["code"])
    assert function_signature, "Function signature could not be extracted."
    if simple_prompt:
        prompt = format_simple_mbpp_prompt(problem, function_signature)
    else:
        prompt = format_custom_mbpp_prompt(problem, function_signature)
    return (prompt, function_signature)


def load_mbpp_problems():
    dataset = load_dataset("mbpp")
    problems = {
        example["task_id"]: example for _, example in enumerate(dataset["test"])
    }
    return problems


def load_jsonl(file_path):
    """Loads a JSON Lines file into a list of dictionaries."""
    data = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                # Remove leading/trailing whitespace and parse
                stripped_line = line.strip()
                if stripped_line:  # Ensure line is not empty
                    try:
                        data.append(json.loads(stripped_line))
                    except json.JSONDecodeError as e:
                        print(
                            f"Skipping line due to JSON decode error: {e} - Line: '{stripped_line}'"
                        )
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    return data


def load_official_results(model_name):
    official_passed_task_ids = set([])
    official_results = {}
    try:
        official_results_path = OFFICIAL_RESULT_PATH[model_name]
        official_passed_task_ids_path = OFFICIAL_PASSED_TASK_IDS_PATH[model_name]
        with open(official_passed_task_ids_path, "r") as f:
            official_passed_task_ids = set(json.load(f))
        official_results_data = load_jsonl(official_results_path)
        official_results = {}
        for entry in official_results_data:
            task_id = entry["task_id"]
            official_results[task_id] = entry
    except Exception as e:
        print(f"Error loading baseline passed IDs: {e}")
        official_passed_task_ids = set([])
        official_results = {}
    return official_passed_task_ids, official_results
