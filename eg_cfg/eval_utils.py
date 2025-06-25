import json
import subprocess
import tempfile
import time
import traceback
import json
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from consts import *


def run_tests(solution, test_cases, io_flag=False, max_workers=8):
    results = {}

    if solution is None:
        for test_case in test_cases:
            test_case_key = json.dumps(test_case) if io_flag else test_case
            results[test_case_key] = {
                "result": False,
                "time": -1,
                "error": "GenerationError",
            }
        return results

    def run_single_test(test_case):
        test_case_key = json.dumps(test_case) if io_flag else test_case
        try:
            if io_flag:
                eval_result = evaluate_solution_io(solution, test_case, timeout=10)
            else:
                eval_result = evaluate_solution(solution, test_case)
            return test_case_key, eval_result
        except Exception as e:
            tb = traceback.format_exc()
            print(f"Problem executing test case: {test_case}")
            return test_case_key, {
                "result": False,
                "time": -1,
                "error": str(type(e)),
                "tb": tb,
            }

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_case = {executor.submit(run_single_test, tc): tc for tc in test_cases}
        for future in as_completed(future_to_case):
            test_case_key, result = future.result()
            results[test_case_key] = result

    return results


def evaluate_solution_io(code, test_case, timeout=15):
    test_passed = False
    error = None
    expected_stdin, expecte_stdout = test_case
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as raw_stdout_file:
        raw_stdout_path = raw_stdout_file.name
    invocation = "solve()"
    injected_prefix = INJECT_IO_EVAL.format(
        expected_stdin=expected_stdin, stdout_path=raw_stdout_path
    )
    test_code = f"{injected_prefix}\n{code}"
    if not code.strip().endswith(invocation):
        test_code = f"{test_code}\n{invocation}"

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

            test_passed = False
            if result.returncode == 0 and not "Traceback" in result.stderr:
                # test_passed = True
                with open(raw_stdout_path, "r") as raw_stdout_content_f:
                    stdout_content = raw_stdout_content_f.read()
                if stdout_content == expecte_stdout:
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


def evaluate_solution(code, test_case, timeout=10):
    test_passed = False
    error = None
    test_code = f"{code}\n{test_case}"

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
