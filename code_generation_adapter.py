import torch
import black
import re

from collections import OrderedDict
from execution_manager import ExecutionManager
from model_utils import extract_new_tokens, calculate_tokens_length
from mbpp_utils import parse_mbpp_assert_statement
from code_generation_utils import generate_code_solutions

NEAREST_FUTURE_DYNAMIC_SIGNAL_PATTERN = """
# Function:
{function_code}

# Invocation:
{test_case}

# Execution Trace: {trace}
"""

SINGLE_DYNAMIC_SIGNAL_PATTERN = """
# Invocation:
{test_case}

# Execution Trace: {trace}
"""

NEAREST_FUTURE_DYNAMIC_SIGNAL_PROMPT = """
### Runtime Behavior for Candidate Continuations
Below are execution traces from running the response function after appending several possible future continuations. These continuations represent plausible ways the function might continue from its current state. They are not necessarily full solutions—some may be partial, exploratory, or incomplete.

For each candidate continuation, multiple test cases (invocations) were executed to observe its behavior under different inputs. Each entry includes:
- A candidate version of the function
- A specific test case used for invocation
- The resulting execution trace for that test case

These dynamic signals can help you better understand how different plausible continuations behave at runtime, and guide you toward a more accurate solution.

{dynamic_signals}
"""

PARTIAL_EXECUTION_DYNAMIC_SIGNAL_PROMPT = """
### Runtime Behavior up to the Last Valid Line
This trace reflects the actual runtime behavior of the response function executed up to the last syntactically or semantically valid line—before the function was completed. It captures how the current partial implementation behaves, which can provide useful context for continuing the function.

Typically, one or more test cases (invocations) are run against this partial version to observe any runtime behavior, including crashes, exceptions, or intermediate outputs.

Use this information to better understand how the partial function performs so far, and to guide the next steps in completing the function correctly.

{dynamic_signals}
"""

BACKWARD_DYNAMIC_SIGNAL_PROMPT = """
### Runtime Behavior of Invalid Solutions
The following examples show complete function solutions that failed to pass validation. These solutions were tested using assertions, and at least one assertion failed during execution—typically resulting in an AssertionError.

Each entry includes:
- A full function solution that failed at least one test
- A specific test case (assertion) that triggered the failure
- The resulting execution trace

Use this information to recognize and avoid common mistakes in incorrect solutions, and to guide your next attempt toward correct and robust behavior.

{dynamic_signals}
"""

BACKWARD_DYNAMIC_SIGNAL_PATTERN = """
# Invalid Solution:
{function_code}

# Test Case (Assertion):
{test_case}

# Execution Trace:
{trace}
"""

DYNAMIC_SIGNAL_PROMPT_REPLACE_STRING = "### Response:"

DYNAMIC_SIGNAL__PARTIAL_EXECUTION = "PartialExecution"
DYNAMIC_SIGNAL__NEAREST_FUTURE_EXECUTION = "NearestFutureExecution"
DYNAMIC_SIGNAL__BACKWARD = "Backward"

SUPPORTED_DYNAMIC_SIGNALS = (
    DYNAMIC_SIGNAL__PARTIAL_EXECUTION,
    DYNAMIC_SIGNAL__NEAREST_FUTURE_EXECUTION,
    DYNAMIC_SIGNAL__BACKWARD,
)


class CodeGenerationAdapter:
    def __init__(
        self,
        model,
        tokenizer,
        device,
        function_signature,
        test_cases,
        initial_prompt,
        dynamic_signals,
        nearest_future_samples=None,
        nearest_future_lines=None,
        backward_signals=(),
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.function_signature = function_signature
        self.test_cases = test_cases
        self.initial_prompt = initial_prompt
        self.initial_prompt_input_ids_len = calculate_tokens_length(
            tokenizer, initial_prompt
        )
        self.program_executions = OrderedDict()
        self.execution_manager = ExecutionManager(tokenizer, function_signature)
        self.nearest_future_samples = nearest_future_samples
        self.nearest_future_lines = nearest_future_lines
        self.backward_signals = backward_signals

        assert dynamic_signals
        for dynamic_signal in dynamic_signals:
            assert dynamic_signal in SUPPORTED_DYNAMIC_SIGNALS
            assert dynamic_signal in self.dynamic_signal_handlers()
        self.dynamic_signals = dynamic_signals

    @staticmethod
    def dynamic_signal_handlers():
        return {
            DYNAMIC_SIGNAL__PARTIAL_EXECUTION: CodeGenerationAdapter._extract_partial_execution_dynamic_signals,
            DYNAMIC_SIGNAL__NEAREST_FUTURE_EXECUTION: CodeGenerationAdapter._extract_nearest_future_execution_dynamic_signals,
            DYNAMIC_SIGNAL__BACKWARD: CodeGenerationAdapter._extract_backward_dynamic_signals,
        }

    def _extract_backward_dynamic_signals(self, input_ids):
        dynamic_signal_text = ""
        if self.backward_signals:
            dynamic_signals = "\n".join(self.backward_signals)
            dynamic_signal_text = BACKWARD_DYNAMIC_SIGNAL_PROMPT.format(
                dynamic_signals=dynamic_signals
            )
            print(dynamic_signal_text)
        return dynamic_signal_text, ()

    def _extract_nearest_future_execution_dynamic_signals(self, input_ids):
        attention_mask = (input_ids != 0).long()
        inputs = {
            "input_ids": input_ids.clone().to(self.device),
            "attention_mask": attention_mask.to(self.device),
        }
        solutions = generate_code_solutions(
            self.model,
            self.tokenizer,
            self.device,
            prompt=None,
            dsgi_manager=None,
            num_return_sequences=self.nearest_future_samples,
            num_beams=self.nearest_future_samples,
            inputs=inputs,
            nearest_future_lines=self.nearest_future_lines,
            do_sample=True,
            return_full=True,
        )

        new_codes = []
        for output in solutions:
            output = output.unsqueeze(0)
            new_code, _ = extract_new_tokens(self.tokenizer, output, input_ids.shape[1])
            new_codes.append(new_code)

        executable_partial_programs = []
        for new_code in new_codes:
            executable_partial_program_code = (
                self.execution_manager.extract_partial_executable_program(new_code)
            )
            executable_partial_programs.append(executable_partial_program_code)
        executable_partial_programs = list(set(executable_partial_programs))

        for executable_partial_program_code in executable_partial_programs:
            # print(executable_partial_program_code)
            # print()
            if executable_partial_program_code not in self.program_executions:
                self.program_executions[executable_partial_program_code] = (
                    self.execution_manager.execute_test_cases(
                        executable_partial_program_code, self.test_cases
                    )
                )

        dynamic_signals = []
        for executable_partial_program_code in executable_partial_programs:
            for test_case, program_execution in self.program_executions[
                executable_partial_program_code
            ].items():
                trace = program_execution.to_compact_json()
                function_name, args_str, _ = parse_mbpp_assert_statement(test_case)
                innvocation = f"{function_name}{args_str}"
                dynamic_signal = NEAREST_FUTURE_DYNAMIC_SIGNAL_PATTERN.format(
                    function_code=executable_partial_program_code,
                    test_case=innvocation,
                    trace=trace,
                )
                dynamic_signals.append(dynamic_signal)
        dynamic_signals = "\n".join(dynamic_signals)
        dynamic_signal_text = NEAREST_FUTURE_DYNAMIC_SIGNAL_PROMPT.format(
            dynamic_signals=dynamic_signals
        )

        # if the last line ends with : add a pass
        return dynamic_signal_text, ()

    def _extract_partial_execution_dynamic_signals(self, input_ids):
        executable_partial_program_code = self._extract_partial_executions(input_ids)
        dynamic_signals = []
        for test_case, program_execution in self.program_executions[
            executable_partial_program_code
        ].items():
            trace = program_execution.to_compact_json()
            function_name, args_str, _ = parse_mbpp_assert_statement(test_case)
            innvocation = f"{function_name}{args_str}"
            dynamic_signal = SINGLE_DYNAMIC_SIGNAL_PATTERN.format(
                test_case=innvocation, trace=trace
            )
            dynamic_signals.append(dynamic_signal)
        dynamic_signals = "\n".join(dynamic_signals)
        dynamic_signal_text = PARTIAL_EXECUTION_DYNAMIC_SIGNAL_PROMPT.format(
            dynamic_signals=dynamic_signals
        )

        ## Debug purposes only
        new_code, _ = extract_new_tokens(
            self.tokenizer, input_ids, self.initial_prompt_input_ids_len
        )
        debug_data = (executable_partial_program_code, new_code)
        return dynamic_signal_text, debug_data

    def unify_dynamic_signals(self, input_ids, dynamic_signals_text):
        new_code, new_code_tokens = extract_new_tokens(
            self.tokenizer, input_ids, self.initial_prompt_input_ids_len
        )

        unified_dynamic_signal_text = ""
        for dynamic_signal in self.dynamic_signals:
            unified_dynamic_signal_text += dynamic_signals_text[dynamic_signal]
        unified_dynamic_signal_text += f"\n{DYNAMIC_SIGNAL_PROMPT_REPLACE_STRING}"

        # inject to original prompt
        unified_dynamic_signal_prompt = self.initial_prompt.replace(
            DYNAMIC_SIGNAL_PROMPT_REPLACE_STRING, unified_dynamic_signal_text
        )

        unified_dynamic_signal_prompt_tokens = self.tokenizer(
            unified_dynamic_signal_prompt, return_tensors="pt"
        )
        dynamic_signal_input_ids = torch.cat(
            [
                unified_dynamic_signal_prompt_tokens["input_ids"].to(self.device),
                new_code_tokens.clone(),
            ],
            dim=1,
        )
        return dynamic_signal_input_ids

    def extract_dynamic_signal_input_ids(self, input_ids):
        dynamic_signals_text = {}
        debug_data = {}

        for dynamic_signal in self.dynamic_signals:
            dynamic_signals_text[dynamic_signal], debug_data[dynamic_signal] = (
                self.dynamic_signal_handlers()[dynamic_signal](self, input_ids)
            )

        dynamic_signal_input_ids = self.unify_dynamic_signals(
            input_ids, dynamic_signals_text
        )
        return dynamic_signal_input_ids, debug_data[DYNAMIC_SIGNAL__PARTIAL_EXECUTION]

    def _extract_partial_executions(self, input_ids: torch.Tensor) -> str:
        new_code, _ = extract_new_tokens(
            self.tokenizer, input_ids.clone(), self.initial_prompt_input_ids_len
        )

        executable_partial_program_code = (
            self.execution_manager.extract_partial_executable_program(new_code)
        )
        if executable_partial_program_code not in self.program_executions:
            self.program_executions[executable_partial_program_code] = (
                self.execution_manager.execute_test_cases(
                    executable_partial_program_code, self.test_cases
                )
            )

        return executable_partial_program_code
