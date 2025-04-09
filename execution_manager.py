import re
import torch
import black
import tempfile
import subprocess
import os
from traces_dumper.program_execution import ProgramExecution
from code_generation_utils import remove_comments_and_docstrings, is_valid_python
from model_utils import extract_new_tokens
from mbpp_utils import parse_mbpp_assert_statement
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed


class ExecutionManager:
    def __init__(self, tokenizer, function_signature=None):
        self.tokenizer = tokenizer
        self.function_signature = function_signature

    def execute_test_cases(self, executable_code, test_cases, use_assert=False):
        executions = {}
        futures = {}

        def run_test_case(test_case):
            try:
                invocation = test_case
                if not use_assert:
                    function_name, args_str, _ = parse_mbpp_assert_statement(test_case)
                    invocation = f"{function_name}{args_str}"
                test_case_code = f"{executable_code}\n{invocation}"
                test_case_code = black.format_str(
                    test_case_code, mode=black.FileMode(line_length=1024)
                )
                assert is_valid_python(
                    test_case_code
                ), f"Invalid Test Case: {test_case}"
                return test_case, self.execute(test_case_code)
            except Exception as e:
                traceback.print_exc()
                print(f"Error in test case: {test_case}")
                return test_case, None

        # Parallel execution using ThreadPoolExecutor
        original_cwd = os.getcwd()
        traces_dumper_dir = os.path.join(original_cwd, "traces_dumper")
        os.chdir(traces_dumper_dir)

        with ThreadPoolExecutor(max_workers=min(8, len(test_cases))) as executor:
            for test_case in test_cases:
                futures[executor.submit(run_test_case, test_case)] = test_case

            for future in as_completed(futures):
                test_case, program_execution = future.result()
                if program_execution is not None:
                    executions[test_case] = program_execution

        os.chdir(original_cwd)
        return executions

    def extract_partial_executable_program(self, new_code) -> str:
        partial_program_code = new_code
        partial_program_code = partial_program_code.replace("```", "")
        if self.function_signature:
            partial_program_code = f"{self.function_signature}\n{new_code}"
        executable_partial_program_code = self.make_executable(partial_program_code)
        executable_partial_program_code = remove_comments_and_docstrings(
            executable_partial_program_code
        )
        executable_partial_program_code = executable_partial_program_code.strip()
        return executable_partial_program_code

    def make_executable(
        self, partial_program_code: str, fallback_to_prompt: bool = True
    ) -> str:
        function_signature = self.function_signature
        lines = partial_program_code.split("\n")
        executable_code = ""

        while lines:
            executable_code = "\n".join(lines)
            is_valid_code = is_valid_python(executable_code) and (
                function_signature is None
                or (executable_code.startswith(function_signature))
            )
            if is_valid_code:
                break

            # Try inserting a pass maybe it will help
            last_line = lines[-1]
            indent = re.match(r"\s*", last_line).group(0)
            lines.append(f"{indent}pass")

            executable_code = "\n".join(lines)
            is_valid_code = is_valid_python(executable_code) and (
                function_signature is None
                or (executable_code.startswith(function_signature))
            )
            if is_valid_code:
                break

            lines.pop()  # remove the pass we added

            # Maybe the last line is problematic.
            last_line = lines.pop()
            if not lines:
                break  # Stop if there are no lines left

            executable_code = "\n".join(lines)
            is_valid_code = is_valid_python(executable_code) and (
                function_signature is None
                or (executable_code.startswith(function_signature))
            )
            if is_valid_code:
                break

            # If removing doesn't work, replace last line with 'pass' (preserving indentation)
            indent = re.match(r"\s*", last_line).group(0)
            lines.append(f"{indent}pass")

            executable_code = "\n".join(lines)
            is_valid_code = is_valid_python(executable_code) and (
                function_signature is None
                or (executable_code.startswith(function_signature))
            )
            if is_valid_code:
                break
            lines.pop()  # Remove the pass if it's still invalid

        is_valid_code = is_valid_python(executable_code) and (
            function_signature is None
            or (executable_code.startswith(function_signature))
        )
        if (not is_valid_code) and fallback_to_prompt:
            if function_signature is not None:
                prompt_lines = function_signature.split("\n")
                last_line = prompt_lines[-1]
                indent = re.match(r"\s*", last_line).group(0)
                if not indent:
                    indent = "   "
                executable_code = f"{function_signature}\n{indent}pass"
                executable_code = black.format_str(
                    executable_code, mode=black.FileMode(line_length=1024)
                )
            else:
                raise ValueError("Not Executable Code to extract")
        return executable_code

    def execute(self, code: str):
        # Step 1: Write the provided code to a temporary Python file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as program_file:
            program_file.write(code)
            program_path = program_file.name

        try:
            # Step 2: Create a temporary file for the raw trace output
            with tempfile.NamedTemporaryFile(mode="w+", delete=False) as raw_trace_file:
                raw_trace_path = raw_trace_file.name

            # Step 3: Run runner.py with the program path, redirecting output to raw trace file
            with open(raw_trace_path, "w") as raw_out:
                subprocess.run(
                    ["python", "runner.py", program_path],
                    stdout=raw_out,
                    check=True,
                    timeout=15,
                )

            # Step 4: Create a temporary file for the formatted trace
            with tempfile.NamedTemporaryFile(
                mode="w+", delete=False
            ) as formatted_trace_file:
                formatted_trace_path = formatted_trace_file.name

            # Step 5: Run formatter.py with the raw trace path, capturing formatted output
            with open(formatted_trace_path, "w") as formatted_out:
                subprocess.run(
                    ["python", "formater.py", raw_trace_path],
                    stdout=formatted_out,
                    check=True,
                )

            # Step 6: Read the contents of both output files
            with open(raw_trace_path, "r") as f:
                raw_trace_content = f.read()

            with open(formatted_trace_path, "r") as f:
                formatted_trace_content = f.read()

            # Step 7: Create and return a ProgramExecution object
            program_execution = ProgramExecution(formatted_trace_path, program_path)

            return program_execution

        finally:
            # Optional: Cleanup logic if needed
            try:
                os.remove(raw_trace_path)
            except:
                pass
            try:
                os.remove(formatted_trace_path)
            except:
                pass
