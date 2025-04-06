import torch
import black
import re

from collections import OrderedDict
from execution_manager import ExecutionManager
from model_utils import extract_new_tokens, calculate_tokens_length
from mbpp_utils import parse_mbpp_assert_statement
from code_generation_utils import (
    generate_code_solutions,
    raw_outputs_to_new_code,
    slice_prompt_after_markers,
)
from consts import *


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
        prompt_type,
        nearest_future_samples=None,
        temperature=None,
        max_function_body_lines=None,
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
        self.prompt_type = prompt_type
        self.program_executions = OrderedDict()
        self.execution_manager = ExecutionManager(tokenizer, function_signature)
        self.nearest_future_samples = nearest_future_samples
        self.temperature = temperature
        self.max_function_body_lines = max_function_body_lines
        self.backward_signals = backward_signals

        assert dynamic_signals
        for dynamic_signal in dynamic_signals:
            assert dynamic_signal in SUPPORTED_DYNAMIC_SIGNALS
            assert dynamic_signal in self.dynamic_signal_handlers()
        self.dynamic_signals = dynamic_signals
        self.detector = None

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
        return dynamic_signal_text, ()

    def _extract_nearest_future_execution_dynamic_signals(self, input_ids):
        if self.detector.function_start_idx is None:
            return "", ()

        attention_mask = (input_ids != 0).long()
        inputs = {
            "input_ids": input_ids.clone().to(self.device),
            "attention_mask": attention_mask.to(self.device),
        }
        function_name, args_str, _ = parse_mbpp_assert_statement(self.test_cases[0])
        outputs = generate_code_solutions(
            self.model,
            self.tokenizer,
            self.device,
            prompt=None,
            dsgi_manager=None,
            num_return_sequences=self.nearest_future_samples,
            temperature=self.temperature,
            inputs=inputs,
            max_function_body_lines=self.max_function_body_lines,
            function_name=function_name,
            do_sample=True,
            prompt_type=self.prompt_type,
        )
        new_codes = raw_outputs_to_new_code(
            outputs,
            self.tokenizer,
            self.initial_prompt_input_ids_len,
            self.prompt_type,
            validate=False,
        )
        new_codes = list(set(new_codes))
        executable_partial_programs = []
        for idx, new_code in enumerate(new_codes):
            try:
                executable_partial_program_code = (
                    self.execution_manager.extract_partial_executable_program(new_code)
                )
            except ValueError:
                continue
            executable_partial_programs.append(executable_partial_program_code)
        executable_partial_programs = list(set(executable_partial_programs))

        for idx, executable_partial_program_code in enumerate(
            executable_partial_programs
        ):
            print(f"#{idx + 1} Executing:\n {executable_partial_program_code}")
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

        dynamic_signal_text = ""
        if dynamic_signals:
            dynamic_signals = "\n".join(dynamic_signals)
            dynamic_signal_text = NEAREST_FUTURE_DYNAMIC_SIGNAL_PROMPT.format(
                dynamic_signals=dynamic_signals
            )

        # if the last line ends with : add a pass
        return dynamic_signal_text, ()

    def _extract_partial_execution_dynamic_signals(self, input_ids):
        executable_partial_program_code = ""
        try:
            executable_partial_program_code = self._extract_partial_executions(
                input_ids
            )
        except ValueError:
            pass

        dynamic_signals = []
        if executable_partial_program_code:
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

        dynamic_signal_text = ""
        if dynamic_signals:
            dynamic_signals = "\n".join(dynamic_signals)
            dynamic_signal_text = PARTIAL_EXECUTION_DYNAMIC_SIGNAL_PROMPT.format(
                dynamic_signals=dynamic_signals
            )

        ## Debug purposes only
        crop_idx = self.initial_prompt_input_ids_len
        if self.detector is not None:
            crop_idx = self.detector.function_start_idx
        new_code, _ = extract_new_tokens(self.tokenizer, input_ids.clone(), crop_idx)
        debug_data = (executable_partial_program_code, new_code)
        return dynamic_signal_text, debug_data

    def unify_dynamic_signals(self, input_ids, dynamic_signals_text):
        new_code, new_code_tokens = extract_new_tokens(
            self.tokenizer, input_ids, self.initial_prompt_input_ids_len
        )

        unified_dynamic_signal_text = ""
        for dynamic_signal in self.dynamic_signals:
            unified_dynamic_signal_text += dynamic_signals_text[dynamic_signal]

        unified_dynamic_signal_prompt = self.initial_prompt
        if unified_dynamic_signal_text:
            if self.prompt_type == PROMPT_TYPE__DEEPSEEK_INSTRUCT:
                unified_dynamic_signal_text += (
                    f"\n{DYNAMIC_SIGNAL_PROMPT_REPLACE_STRING_INSTRUCT_BEGIN}"
                )
                unified_dynamic_signal_prompt = self.initial_prompt.replace(
                    DYNAMIC_SIGNAL_PROMPT_REPLACE_STRING_INSTRUCT_BEGIN,
                    unified_dynamic_signal_text,
                )
            if self.prompt_type == PROMPT_TYPE__DEEPSEEK_BASE:
                begin_idx = self.initial_prompt.find(
                    DYNAMIC_SIGNAL_PROMPT_REPLACE_STRING_BASE_BEGIN
                )
                if begin_idx != -1:
                    injection = unified_dynamic_signal_text
                    modified_prompt = (
                        self.initial_prompt[:begin_idx]
                        + injection
                        + self.initial_prompt[begin_idx:]
                    )
                else:
                    modified_prompt = self.initial_prompt
                unified_dynamic_signal_prompt = modified_prompt

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
        return dynamic_signal_input_ids, debug_data.get(
            DYNAMIC_SIGNAL__PARTIAL_EXECUTION, ("", "")
        )

    def _extract_partial_executions(self, input_ids: torch.Tensor) -> str:
        crop_idx = self.initial_prompt_input_ids_len
        new_code, _ = extract_new_tokens(self.tokenizer, input_ids.clone(), crop_idx)
        if self.prompt_type == PROMPT_TYPE__DEEPSEEK_BASE:
            # Nothing to do, everything after [BEGIN] should be code
            pass
        if self.prompt_type == PROMPT_TYPE__DEEPSEEK_INSTRUCT:
            new_code = slice_prompt_after_markers(
                new_code, marker=INSTRUCT_MODEL_PYTHON_CODE_START
            )

        executable_partial_program_code = (
            self.execution_manager.extract_partial_executable_program(new_code)
        )
        if (
            executable_partial_program_code
            and executable_partial_program_code not in self.program_executions
        ):
            print(f"Executing:\n {executable_partial_program_code}")
            self.program_executions[executable_partial_program_code] = (
                self.execution_manager.execute_test_cases(
                    executable_partial_program_code, self.test_cases
                )
            )

        return executable_partial_program_code
