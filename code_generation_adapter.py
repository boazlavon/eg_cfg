import torch
import black
import re

from collections import OrderedDict
from execution_manager import ExecutionManager
from model_utils import extract_new_tokens, calculate_tokens_length
from mbpp_utils import parse_mbpp_assert_statement

SINGLE_DYNAMIC_SIGNAL_PATTERN = """
# Test Case Invocation:
{test_case}

# Execution Trace: {trace}
"""

DYNAMIC_SIGNAL_PROMPT = """
### Additional Data on the Runtime Behavior
This trace reflects the actual runtime behavior of the response function up to the last valid line and is intended to provide helpful context for predicting the next token in the function’s completion.
This information may help refine your solution by revealing potential issues or confirming expected behavior during execution.

{dynamic_signals}

### Response:
"""

DYNAMIC_SIGNAL_PROMPT_REPLACE_STRING = "### Response:"


class CodeGenerationAdapter:
    def __init__(
        self, tokenizer, device, function_signature, test_cases, initial_prompt
    ):
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

    def extract_dynamic_signal_input_ids(self, input_ids):
        executable_partial_program_code = self._extract_partial_executions(input_ids)
        new_code, new_code_tokens = extract_new_tokens(
            self.tokenizer, input_ids, self.initial_prompt_input_ids_len
        )
        dynamic_signal_input_ids = self._inject_dynamic_signals(
            executable_partial_program_code, new_code_tokens
        )

        debug_data = executable_partial_program_code, new_code
        return dynamic_signal_input_ids, debug_data

    def _extract_partial_executions(self, input_ids: torch.Tensor) -> str:
        executable_partial_program_code = (
            self.execution_manager.extract_partial_executable_program(
                input_ids.clone(), self.initial_prompt_input_ids_len
            )
        )
        if executable_partial_program_code not in self.program_executions:
            self.program_executions[executable_partial_program_code] = (
                self.execution_manager.execute_test_cases(
                    executable_partial_program_code, self.test_cases
                )
            )

        return executable_partial_program_code

    def _inject_dynamic_signals(self, executable_partial_program_code, new_code_tokens):
        unified_dynamic_signal_prompt = self._to_dynamic_signal_prompt(
            self.program_executions[executable_partial_program_code]
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

    def _to_dynamic_signal_prompt(self, executions):
        dynamic_signals = []
        for test_case, program_execution in executions.items():
            trace = program_execution.to_compact_json()
            function_name, args_str, _ = parse_mbpp_assert_statement(test_case)
            innvocation = f"{function_name}{args_str}"
            dynamic_signal = SINGLE_DYNAMIC_SIGNAL_PATTERN.format(
                test_case=innvocation, trace=trace
            )
            dynamic_signals.append(dynamic_signal)
        dynamic_signals = "\n".join(dynamic_signals)
        dynamic_signal_prompt = DYNAMIC_SIGNAL_PROMPT.format(
            dynamic_signals=dynamic_signals
        )
        unified_dynamic_signal_prompt = self.initial_prompt.replace(
            DYNAMIC_SIGNAL_PROMPT_REPLACE_STRING, dynamic_signal_prompt
        )
        unified_dynamic_signal_prompt = unified_dynamic_signal_prompt.replace(
            "Response:\n\n", "Response:\n"
        )
        return unified_dynamic_signal_prompt
