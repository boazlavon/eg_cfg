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
    return (a + b) * (c / (a - b) + (a * c) / (b + c))  # Too complex to read

Correct:
def compute_value(a, b, c):
    term1 = a + b  # Compute first term separately
    term2 = a - b  # Store denominator separately
    term3 = c / term2  # Compute first fraction
    term4 = a * c / (b + c)  # Compute second fraction
    result = term1 * (term3 + term4)  # Combine step by step
    return result

### Response:
{function_signature}

"""


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

    correct = 0
    incorrect = 0
    total = 0
    errors = []
    timeouts = []
    failed_tests = {}
    passed_tests = []

    dataset = load_dataset("mbpp")
    problems = {
        example["task_id"]: example for _, example in enumerate(dataset["test"])
    }

    for task_id, code in task_functions.items():
        task = problems[task_id]
        test_cases = task["test_list"]

        all_passed = True
        failed_tests_task = []

        for i, test_case in enumerate(test_cases):
            test_code = f"{code}\n{test_case}"
            test_code = black.format_str(
                test_code, mode=black.FileMode(line_length=1024)
            )

            with tempfile.NamedTemporaryFile(suffix=".py", mode="w") as temp_file:
                temp_file.write(test_code)
                temp_file.flush()

                # start_time = time.time()
                try:
                    result = subprocess.run(
                        ["python", temp_file.name],
                        capture_output=True,
                        text=True,
                        timeout=timeout,
                    )
                    # end_time = time.time()

                    if result.returncode != 0:
                        all_passed = False
                        failed_tests_task.append(i)
                    elif "Traceback" in result.stderr:
                        all_passed = False
                        failed_tests_task.append(i)

                except subprocess.TimeoutExpired:
                    all_passed = False
                    timeouts.append(task_id)
                    break  # No need to test further if it timed out.
                except Exception as e:
                    all_passed = False
                    errors.append(task_id)
                    print(f"Error executing test for task {task_id}: {e}")
                    break

        if all_passed:
            correct += 1
            passed_tests.append(task_id)
        elif failed_tests_task:
            incorrect += 1
            failed_tests[task_id] = failed_tests_task
        total += 1

        accuracy = correct / total if total > 0 else 0
        results_dict = {
            "total": total,
            "correct": correct,
            "incorrect": incorrect,
            "accuracy": accuracy,
            "errors": errors,
            "timeouts": timeouts,
            "failed_tests": failed_tests,
            "passed_tests": passed_tests,
        }
        print(task_id)
        print(results_dict)

    return results_dict


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
