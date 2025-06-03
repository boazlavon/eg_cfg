import re
import json
import subprocess
import tempfile
import time
import black
import traceback
from datasets import load_dataset
from collections import OrderedDict
from consts import *

TEST_CASES_INSTRUCTION = """
Write a Python function that satisfies the following test cases:
>>> Test Cases:
{test_cases}
"""

CUSTOM_INSTRUCTION_TEXT = """\
You are an AI programming assistant, utilizing the Deepseek Coder model, developed by Deepseek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer
### Instruction:
{problem_text}

Write a Python function that satisfies the following test cases:
>>> Test Cases:
{test_cases}

Your solution should be written in as many lines as possible.
This ensures that prefixes of your function remain valid Python programs.
Allowing **incremental execution and debugging**.

Write the function **step by step**, progressively introducing variables and logic.
Avoid using list comprehensions, lambda functions, or overly compact one-liners.
Instead, follow these guidelines:**

Avoid list comprehensions, use loops instead:
Incorrect:
```python
def square_numbers(lst):
    return [x ** 2 for x in lst]
```

Correct:
```python
def square_numbers(lst):
    squares = []
    for num in lst:
        squared_value = num ** 2
        squares.append(squared_value)
    return squares
```

Avoid inline expressions, use variables instead
Incorrect:
```python
def calculate_area(length, width):
    return (length * width) / 2
```

Correct:
```python
def calculate_area(length, width):
    product = length * width
    area = product / 2
    return area
```

Incorrect:
```python
result.append(x + y)
```

Correct:
```python
z = x + y
result.append(z)
```

Incorrect:
```python
def compute_value(a, b, c):
    return (a + b) * (c / (a - b) + (a * c) / (b + c))
```

Correct:
```python
def compute_value(a, b, c):
    term1 = a + b 
    term2 = a - b 
    term3 = c / term2 
    term4 = a * c / (b + c)
    result = term1 * (term3 + term4)
    return result
```

### Response:
"""

DEEPSEEK_INSTRUCT_TESTCASES_INSTRUCTION_TMP = """
>>>> Test Cases:
{test_cases}
"""

DEEPSEEK_INSTRUCT_TESTCASES_INSTRUCTION = """
>>> Test Cases:
{test_cases}
"""

DEEPSEEK_INSTRUCT_TEMPLATE = """\
You are an AI programming assistant, utilizing the Deepseek Coder model, developed by Deepseek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer
### Instruction:
Please refer the given examples and generate a python function for my problem.
Examples are listed as follows:

- Example 1:
>>> Problem:
Write a function to find the similar elements from the given two tuple lists.
>>> Test Cases:
assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)
assert similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4)
assert similar_elements((11, 12, 14, 13),(17, 15, 14, 13)) == (13, 14)

>>> Code:
```python
def similar_elements(test_tup1, test_tup2):
  res = tuple(set(test_tup1) & set(test_tup2))
  return (res)
```

- Example 2:
>>> Problem:
Write a python function to identify non-prime numbers.
>>> Test Cases:
assert is_not_prime(2) == False
assert is_not_prime(10) == True
assert is_not_prime(35) == True

>>> Code:
```python
import math
def is_not_prime(n):
    result = False
    for i in range(2,int(math.sqrt(n)) + 1):
        if n % i == 0:
            result = True
    return result
```

- Example 3:
>>> Problem:
Write a function to find the largest integers from a given list of numbers using heap queue algorithm.
>>> Test Cases:
assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],3)==[85, 75, 65]
assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],2)==[85, 75]
assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],5)==[85, 75, 65, 58, 35]

>>> Code:
```python
import heapq as hq
def heap_queue_largest(nums,n):
  largest_nums = hq.nlargest(n, nums)
  return largest_nums
```

Here is my problem:
>>> Problem:
{problem_text}
>>>> Test Cases:
{test_cases}

### Response:
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
    # test_code = black.format_str(test_code, mode=black.FileMode(line_length=1024))

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


def format_custom_mbpp_prompt(problem):
    test_cases = "\n".join(problem["test_list"])
    prompt_template = CUSTOM_INSTRUCTION_TEXT
    if test_cases:
        prompt = prompt_template.format(
            problem_text=problem["text"],
            test_cases=test_cases,
        )
    else:
        prompt_template = prompt_template.replace(TEST_CASES_INSTRUCTION, "")
        prompt = prompt_template.format(
            problem_text=problem["text"],
        )
    return prompt


def format_deepseek_instruct_mbpp_prompt(problem):
    test_cases = "\n".join(problem["test_list"])
    prompt_template = DEEPSEEK_INSTRUCT_TEMPLATE
    if test_cases:
        prompt = prompt_template.format(
            problem_text=problem["text"],
            test_cases=test_cases,
        )
        prompt = prompt.replace(">>>>", ">>>")
    else:
        prompt_template = prompt_template.replace(
            DEEPSEEK_INSTRUCT_TESTCASES_INSTRUCTION_TMP, ""
        )
        prompt = prompt_template.format(
            problem_text=problem["text"],
        )
    return prompt


def format_mbpp_prompt(problem, deepseek_instruct=False):
    function_signature = None
    if deepseek_instruct:
        prompt = format_deepseek_instruct_mbpp_prompt(problem)
    # if simple_prompt:
    #     function_signature = extract_function_signature(problem["code"])
    #     assert function_signature, "Function signature could not be extracted."
    #     prompt = format_simple_mbpp_prompt(problem, function_signature)
    else:
        prompt = format_custom_mbpp_prompt(problem)
    return (prompt, function_signature)


def load_mbpp_problems():
    test_ds = load_dataset("google-research-datasets/mbpp", "full", split="test")
    problems = OrderedDict((example["task_id"], example) for example in test_ds)
    return problems


def extract_asserts_for_candidate_function(test_string: str) -> list[str]:
    assert_statements = []

    # First, locate the 'def check(candidate):' block
    # This regex captures everything inside the check function, up to the next top-level statement or end of string.
    check_function_pattern = re.compile(
        r"def check\(candidate\):\s*\n(.*?)(?=\n[A-Za-z_]|\Z)", re.DOTALL
    )
    match = check_function_pattern.search(test_string)

    if match:
        check_body = match.group(1)
        # Now, find all assert statements within that body that explicitly call 'candidate('
        # ^\s*assert\s+          -> Start of line, optional leading whitespace, 'assert', one or more spaces
        # candidate\s*\(         -> The literal 'candidate' followed by optional whitespace and an opening parenthesis
        # .*$                    -> Match any characters until the end of the line
        assert_pattern = re.compile(r"^\s*assert\s+candidate\s*\(.*$", re.MULTILINE)
        found_asserts = assert_pattern.findall(check_body)

        # Strip trailing whitespace from each found assert statement
        assert_statements = [stmt.strip() for stmt in found_asserts]

    return assert_statements


HUMANEVAL_INSTRUCTION_TEMPLATE = """
Write a function that performs the following task: {instruction}
It should have the following function signature: {function_signature}
"""


def test_case_to_assert(invocation: str, expected: str) -> str:
    """Convert a (input, expected_output) pair to a Python assert statement."""
    return f"assert {invocation} == {expected}"


def load_humaneval_problems():
    with open("data/humaneval/humaneval.json", "r") as f:
        test_ds = json.load(f)
    problems = OrderedDict()
    for task_id, example in test_ds.items():
        new_example = dict(example)
        eval_test_list = extract_asserts_for_candidate_function(example["test"])
        function_signature = example["function_signature"][4:].strip()
        instruction_text = HUMANEVAL_INSTRUCTION_TEMPLATE.format(
            instruction=example["text"], function_signature=function_signature
        ).strip()
        test_cases = [
            test_case_to_assert(invocation, expected)
            for (invocation, expected) in example["test_list"]
        ]
        function_name = example["entry_point"]
        eval_tests = []
        for eval_test in eval_test_list:
            eval_test = eval_test.replace("candidate", function_name)
            eval_tests.append(eval_test)

        raw_test = example["test"]
        raw_test = f"{raw_test}\ncheck({function_name})"
        raw_test = [raw_test]

        new_example = {
            "task_id": task_id,
            "text": instruction_text,
            "code": example["code"],
            "test_list": test_cases,
            "eval_test_list": raw_test,
            # "eval_test_list": eval_tests,
            "entry_point": function_name,
        }
        problems[task_id] = new_example

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
    official_passed_task_ids = list(official_passed_task_ids)
    return official_passed_task_ids, official_results
