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

# DYNAMIC_SIGNAL_PROMPT = """
# ### Additional Data on the Runtime Behavior
# This trace reflects the actual runtime behavior of the response function up to the last valid line and is intended to provide helpful context for predicting the next token in the function’s completion.
# This information may help refine your solution by revealing potential issues or confirming expected behavior during execution.

# {dynamic_signals}

# ### Response:
# """

PARTIAL_EXECUTION_DYNAMIC_SIGNAL_PROMPT = """
### Additional Data on the Runtime Behavior
This trace reflects the actual runtime behavior of the response function up to the last valid line and is intended to provide helpful context for predicting the next token in the function’s completion.
This information may help refine your solution by revealing potential issues or confirming expected behavior during execution.

{dynamic_signals}
"""

DYNAMIC_SIGNAL_PROMPT_REPLACE_STRING = "### Response:"

DYNAMIC_SIGNAL__PARTIAL_EXECUTION = "PartialExecution"
SUPPORTED_DYNAMIC_SIGNALS = (DYNAMIC_SIGNAL__PARTIAL_EXECUTION,)


class CodeGenerationAdapter:
    def __init__(
        self,
        tokenizer,
        device,
        function_signature,
        test_cases,
        initial_prompt,
        dynamic_signals,
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

        assert dynamic_signals
        for dynamic_signal in dynamic_signals:
            assert dynamic_signal in SUPPORTED_DYNAMIC_SIGNALS
            assert dynamic_signal in self.dynamic_signal_handlers()
        self.dynamic_signals = dynamic_signals

    @staticmethod
    def dynamic_signal_handlers():
        return {
            DYNAMIC_SIGNAL__PARTIAL_EXECUTION: CodeGenerationAdapter._extract_partial_execution_dynamic_signals,
        }

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
        for dynamic_signal in SUPPORTED_DYNAMIC_SIGNALS:
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
