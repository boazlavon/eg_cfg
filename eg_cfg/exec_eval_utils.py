from enum import Enum
import requests
import json
from consts import *

###############################################################
### ExecEval Repository: https://github.com/ntunlp/ExecEval ###
###############################################################


# ExecEval/eval_scripts/exec_outcome.py
class ExecEval__ExecOutcome(Enum):
    """
    Enum representing execution outcomes:
    - PASSED: Code executed and output matched expected output
    - WRONG_ANSWER: Code executed but output did not match
    - TIME_LIMIT_EXCEEDED: Code ran too long
    - RUNTIME_ERROR: Code crashed during execution
    - COMPILATION_ERROR: Code failed to compile
    - MEMORY_LIMIT_EXCEEDED: Code used too much memory
    """

    PASSED = "PASSED"
    WRONG_ANSWER = "WRONG_ANSWER"
    TIME_LIMIT_EXCEEDED = "TIME_LIMIT_EXCEEDED"
    RUNTIME_ERROR = "RUNTIME_ERROR"
    COMPILATION_ERROR = "COMPILATION_ERROR"
    MEMORY_LIMIT_EXCEEDED = "MEMORY_LIMIT_EXCEEDED"


# ExecEval/eval_scripts/api_comm.py
class ExecEval__APICommunication:
    def __init__(self, exec_eval_host_ip, exec_eval_host_port):
        assert exec_eval_host_ip is not None, "ExecEval host IP must be specified"
        assert exec_eval_host_port is not None, "ExecEval host port must be specified"
        self.session = requests.Session()
        self.execute_code_url = EXEC_EVAL__EXECUTE_CODE_URL_TEMPLATE.format(
            exec_eval_host_ip=exec_eval_host_ip, exec_eval_host_port=exec_eval_host_port
        )
        self.test_connection()

    def test_connection(self):
        source_code = "def solve():\n    pass\nsolve()"
        unittests = [{"input": "", "output": [""]}]
        results = self.execute_tests(source_code, unittests)
        assert (
            results[0].get("exec_outcome") == ExecEval__ExecOutcome.PASSED.value
        ), "ExecEval session initialization failed, expected PASSED outcome, got: "

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.session.close()

    def execute_tests(
        self,
        source_code,
        unittests,
    ):
        assert source_code is not None, "Source code must be specified"
        assert (
            unittests is not None and len(unittests) > 0
        ), "Unittests must be specified"
        request_body = {
            "language": EXEC_EVAL__LANGUAGE__PYTHON3,
            "source_code": source_code,
            "unittests": unittests,
            "limits": EXEC_EVAL__LANGUAGE__PYTHON3__LIMITS,
            "compile_cmd": None,
            "compile_flags": None,
            "execute_cmd": None,
            "execute_flags": None,
            "block_network": None,
            "stop_on_first_fail": None,
            "use_sanitizer": False,
        }

        response = self.session.post(
            self.execute_code_url,
            json=request_body,
            headers=EXEC_EVAL__REQUEST_HEADERS,
        )
        json_response = response.json()
        response_data = json_response.get("data")
        return response_data


def contest_evaluate(exec_eval_session, generated_code, tests):
    eval_results = exec_eval_session.execute_tests(
        generated_code,
        tests,
    )
    assert eval_results is not None, "Eval results must not be None"

    output_results = {}
    for result in eval_results:
        io_pair = (result["input"], result["output"])
        io_pair = json.dumps(io_pair)
        actual_output = result["result"]
        test_passed = result["exec_outcome"] == ExecEval__ExecOutcome.PASSED.value
        entry = {
            "result": test_passed,
            "time": result.get("time_consumed", None),
            "error": None,
            "actual_output": actual_output,
        }
        output_results[io_pair] = entry

    return output_results


def convert_to_dict_format(samples):
    return [{"input": inp, "output": [out]} for inp, out in samples]


def exec_eval__run_tests(code, test_cases, exec_eval_session):
    dict_test_cases = convert_to_dict_format(test_cases)
    invocation = CODE_CONTESTS__INVOCATION
    test_code = code
    if not code.strip().endswith(invocation):
        test_code = f"{test_code}\n{invocation}"
    output_results = contest_evaluate(
        exec_eval_session,
        test_code,
        dict_test_cases,
    )
    correct = sum(int(entry.get("result", False)) for entry in output_results.values())
    total = len(output_results)
    accuracy = correct / total if total else 0.0
    passed = accuracy == 1.0
    solution_entry = {
        "code": test_code,
        "results": {},
        "passed": passed,
        "accuracy": accuracy,
        "results": output_results,
    }
    return solution_entry
