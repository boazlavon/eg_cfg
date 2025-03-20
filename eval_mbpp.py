import subprocess
import tempfile
import time
import black
from datasets import load_dataset

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
    problems = {example['task_id']: example for _, example in enumerate(dataset["test"])}

    for task_id, code in task_functions.items():
        task = problems[task_id]
        test_cases = task['test_list']

        all_passed = True
        failed_tests_task = []

        for i, test_case in enumerate(test_cases):
            test_code = f"{code}\n{test_case}"
            test_code = black.format_str(test_code, mode=black.FileMode(line_length=1024))

            with tempfile.NamedTemporaryFile(suffix='.py', mode='w') as temp_file:
                temp_file.write(test_code)
                temp_file.flush()

                start_time = time.time()
                try:
                    result = subprocess.run(['python', temp_file.name], capture_output=True, text=True, timeout=timeout)
                    end_time = time.time()

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
        total+=1
        
        accuracy = correct / total if total > 0 else 0
        results_dict = {
            'total': total,
            'correct': correct,
            'incorrect': incorrect,
            'accuracy': accuracy,
            'errors': errors,
            'timeouts': timeouts,
            'failed_tests': failed_tests,
            'passed_tests': passed_tests,
        }
        print(task_id)
        print(results_dict)

    return results_dict

if __name__ == "__main__":
    # Example usage with a dictionary:
    task_functions = {
        "1": "def add_numbers(a, b):\n  return a + b",
        "2": "def multiply_numbers(a, b):\n  return a * b",
        # Add more task IDs and function code strings here
    }

    results = evaluate_mbpp_from_dict(task_functions)
    import json
    print(json.dumps(results, indent=4))