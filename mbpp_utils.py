import re
import subprocess
import tempfile
import time
import black
from datasets import load_dataset

INSTRUCTION_TEXT = """\
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
            # print('Timeout')
            error = "Timeout"
            pass
        except Exception as e:
            # print(f"Error executing test for task")
            error = "Exception"
            pass
        finally:
            end_time = time.time()
            delta_time = end_time - start_time

    result_entry = {"result": test_passed, "time": delta_time, "error": error}
    return result_entry


def evaluate_mbpp_from_dict(task_functions, timeout=10):
    """
    Evaluates LLM predictions on the MBPP benchmark directly from a dictionary,
    using the Hugging Face datasets library.

    Args:
        task_functions (dict): A dictionary where keys are task IDs and values are
                               Python function code strings.
        timeout (int): Timeout in seconds for each test case.

    Returns:
        dict: A dictionary containing evaluation results, including:
              - 'total': Total number of tasks.
              - 'correct': Number of tasks where all test cases passed.
              - 'accuracy': Accuracy (correct / total).
              - 'errors': A list of task IDs where errors occurred.
              - 'timeouts': A list of task IDs where timeouts occurred.
              - 'failed_tests': A dictionary where keys are task IDs and values are lists of failed test case indices.
    """

    errors = []
    timeouts = []
    problems = read_problems()
    results = {}
    # correct = 0
    # incorrect = 0
    # total = 0
    # failed_tests = {}
    # passed_tests = []

    for (task_id, gamma), code in task_functions.items():
        task = problems[task_id]
        test_cases = task["test_list"]

        all_tests_passed = True
        results[(task_id, gamma)] = {}

        for i, test_case in enumerate(test_cases):
            result_entry = evaluate_solution(code, test_case)
            results[(task_id, gamma)] = result_entry
            # all_tests_passed &= test_passed

        # if all_tests_passed:
        #     correct += 1
        #     passed_tests.append(task_id)
        # elif failed_tests_task:
        #     incorrect += 1
        #     failed_tests[task_id] = failed_tests_task
        # total += 1

        # accuracy = correct / total if total > 0 else 0
        # results_dict = {
        #     "total": total,
        #     "correct": correct,
        #     "incorrect": incorrect,
        #     "accuracy": accuracy,
        #     "errors": errors,
        #     "timeouts": timeouts,
        #     "failed_tests": failed_tests,
        #     "passed_tests": passed_tests,
        #     "code", code,
        #     "test_case", test_case
        # }
        # print(task_id)
        # print(results_dict)

    return results


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
        problem_text=problem["text"],
        function_signature=function_signature,
        test_cases=formatted_test_cases,
    )

    return prompt, function_signature


def read_problems():
    dataset = load_dataset("mbpp")
    problems = {
        example["task_id"]: example for _, example in enumerate(dataset["test"])
    }
    return problems
