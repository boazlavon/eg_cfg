import tempfile
import subprocess
import os
from traces_dumper.program_execution import ProgramExecution


class ExecutionManager:
    def execute(self, code: str):
        original_cwd = os.getcwd()
        traces_dumper_dir = os.path.join(original_cwd, "traces_dumper")
        os.chdir(traces_dumper_dir)

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
                    timeout=60,
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

            return program_execution, raw_trace_content, formatted_trace_content

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
            os.chdir(original_cwd)
