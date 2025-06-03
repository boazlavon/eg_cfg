import os
import torch
from collections import OrderedDict
from model_utils import extract_new_tokens, calculate_tokens_length
from mbpp_utils import parse_mbpp_assert_statement
from code_generation_utils import (
    generate_code_solutions,
    raw_outputs_to_new_code,
    slice_prompt_after_markers,
)
from inference_endpoint_utils import inference_endpoint_utils__sample_code_beam_search
from consts import *


class EarlyTerminationExceptin(Exception):
    pass


class CodeGenerationAdapter:
    def __init__(
        self,
        model,
        tokenizer,
        device,
        function_signature,
        test_cases,
        initial_prompt,
        dynamic_signals_types,
        prompt_type,
        use_local_hf_model,
        use_inference_endpoint,
        bs_candidates_count=None,
        temperature=None,
        bs_completion_horizon=None,
        guidance_strategy=None,
        execution_manager=None,
        stats_manager=None,
        model_name=None,
        task_id=None,
        solved_tasks_cache_dir=None,
    ):
        assert (
            use_local_hf_model ^ use_inference_endpoint
        ), "Exactly one of 'use_local_hf_model' or 'use_inference_endpoint' must be True"
        self.use_local_hf_model = use_local_hf_model
        self.use_inference_endpoint = use_inference_endpoint
        if self.use_local_hf_model:
            self.device = device
            self.model = model
        elif self.use_inference_endpoint:
            self.device = None
            self.model = None

        self.tokenizer = tokenizer
        self.function_signature = function_signature
        self.test_cases = test_cases
        self.initial_prompt = initial_prompt
        self.initial_prompt_input_ids_len = calculate_tokens_length(
            tokenizer, initial_prompt
        )
        self.prompt_type = prompt_type
        self.program_executions = OrderedDict()
        self.execution_manager = execution_manager
        self.bs_candidates_count = bs_candidates_count
        self.temperature = temperature
        self.bs_completion_horizon = bs_completion_horizon
        self.guidance_strategy = guidance_strategy
        self.lines_count = 0
        self.current_bs_candidates_count = []
        self.current_dynamic_signal = {}
        self.current_debug_data = {}

        assert dynamic_signals_types
        for dynamic_signal_type in dynamic_signals_types:
            assert dynamic_signal_type in SUPPORTED_DYNAMIC_SIGNALS
            assert dynamic_signal_type in self.dynamic_signal_handlers()
            self.current_dynamic_signal[dynamic_signal_type] = None
            self.current_debug_data[dynamic_signal_type] = None
        self.generate_new_signal = None
        self.dynamic_signals_types = dynamic_signals_types
        self.detector = None
        self.stats_manager = stats_manager
        self.model_name = model_name
        self.task_id = task_id
        self.solved_tasks_cache_dir = solved_tasks_cache_dir

        self.early_stop_detected = False
        self.early_stop_detected_program = None
        self.early_stop_counter = 0
        self.early_stop_threshold = EARLY_STOP_THRESHOLD
        self.perform_early_stop = False

        self.dynamic_early_stop_detected = False
        self.dynamic_early_stop_detected_program = None
        self.dynamic_early_stop_counter = 0
        self.dynamic_early_stop_threshold = EARLY_STOP_THRESHOLD
        self.perform_dynamic_early_stop = False

    @staticmethod
    def dynamic_signal_handlers():
        return {
            DYNAMIC_SIGNAL__PARTIAL_EXECUTION: CodeGenerationAdapter._extract_partial_execution_dynamic_signals,
            DYNAMIC_SIGNAL__MULTIPLE_CANDIDATES_EXECUTION: CodeGenerationAdapter._extract_multiple_candidates_execution_dynamic_signals,
        }

    def query_early_stop(self):
        if self.early_stop_counter >= self.early_stop_threshold * 1.5:
            print("We are in a loop, lets early stop")
            return True
        if self.dynamic_early_stop_counter >= self.dynamic_early_stop_threshold * 1.5:
            print("We are in a loop, lets early stop")
            return True
        if self.perform_early_stop and self.perform_dynamic_early_stop:
            if (
                self.early_stop_detected_program
                == self.dynamic_early_stop_detected_program
            ):
                print("Lets early stop!")
                return True
            else:
                print("Early stop detected program are different!")
                return False
        else:
            if self.early_stop_counter > 0 or self.dynamic_early_stop_counter > 0:
                print(
                    f"Early stop threshold are not met yet {self.dynamic_early_stop_counter}/{self.dynamic_early_stop_threshold}, {self.early_stop_counter}/{self.early_stop_threshold}"
                )
            return False

    def _extract_multiple_candidates_execution_dynamic_signals(
        self, dynamic_signal_type, input_ids
    ):
        unique_stats_entry = []
        unique_stats_entry_raw = []
        new_code = self._extract_new_code(input_ids)
        generate_new_signal = self._do_generate_new_signal(
            dynamic_signal_type, new_code
        )
        self.generate_new_signal = generate_new_signal
        if not generate_new_signal:
            return (
                self.current_dynamic_signal[dynamic_signal_type],
                self.current_debug_data[dynamic_signal_type],
            )

        print("Generate New Signal!")
        self.check_early_termination()
        if self.detector.function_start_idx is None:
            self.current_dynamic_signal[dynamic_signal_type] = ""
            self.current_debug_data[dynamic_signal_type] = ()
            return (
                self.current_dynamic_signal[dynamic_signal_type],
                self.current_debug_data[dynamic_signal_type],
            )

        function_name, args_str, _ = parse_mbpp_assert_statement(self.test_cases[0])
        if self.use_local_hf_model:
            attention_mask = (input_ids != 0).long()
            inputs = {
                "input_ids": input_ids.clone().to(self.device),
                "attention_mask": attention_mask.to(self.device),
            }
            outputs = generate_code_solutions(
                self.model,
                self.tokenizer,
                None,
                inputs,
                num_return_sequences=self.bs_candidates_count,
                temperature=self.temperature,
                bs_completion_horizon=self.bs_completion_horizon,
                function_name=function_name,
                do_sample=True,
                prompt_type=self.prompt_type,
                stats_manager=self.stats_manager,
            )
            new_codes = raw_outputs_to_new_code(
                outputs,
                self.tokenizer,
                self.initial_prompt_input_ids_len,
                self.prompt_type,
                validate=False,
                stats_manager=self.stats_manager,
            )
        elif self.use_inference_endpoint:
            assert self.model_name
            new_codes = inference_endpoint_utils__sample_code_beam_search(
                input_ids,
                tokenizer=self.tokenizer,
                execution_manager=self.execution_manager,
                stats_manager=self.stats_manager,
                candidates_count=self.bs_candidates_count,
                temperature=self.temperature,
                bs_completion_horizon=self.bs_completion_horizon,
                model_name=self.model_name,
                prompt_with_cot=self.prompt_with_cot,
            )
        new_codes = list(set(new_codes))
        unique_stats_entry.append(len(new_codes))
        unique_stats_entry_raw.append(new_codes)
        executable_partial_programs = []

        # print(f"New Codes: {len(new_codes)}")
        for idx, new_code in enumerate(new_codes):
            try:
                # print(f"#{idx + 1} Extracting Partial Executable")
                executable_partial_program_code = (
                    self.execution_manager.extract_partial_executable_program(new_code)
                )
            except ValueError:
                # print(f"#{idx + 1} Error Extracting Partial Executable")
                continue
            executable_partial_programs.append(executable_partial_program_code)
        unique_stats_entry.append(len(executable_partial_programs))
        unique_stats_entry_raw.append(executable_partial_programs)
        executable_partial_programs = list(set(executable_partial_programs))

        # print(f"Executable Programs: {len(executable_partial_programs)}")
        if executable_partial_programs:
            self.current_bs_candidates_count = executable_partial_programs

        for idx, executable_partial_program_code in enumerate(
            executable_partial_programs
        ):
            print(f"#{idx + 1} Executing:\n {executable_partial_program_code}\n")
            if executable_partial_program_code not in self.program_executions:
                self.program_executions[executable_partial_program_code] = (
                    self.execution_manager.execute_test_cases(
                        executable_partial_program_code, self.test_cases
                    )
                )

        if self.early_stop_detected and len(executable_partial_programs) != 1:
            print(
                f"[UNCOND] Cancel Early Stop since no unique executables ({self.early_stop_counter}/{self.early_stop_threshold})"
            )
            self.early_stop_detected = False
            self.early_stop_counter = 0
            self.early_stop_detected_program = None

        if len(executable_partial_programs) == 1:
            unique_program = executable_partial_programs[0].strip()
            if self.early_stop_detected:
                if self.early_stop_detected_program == unique_program:
                    # We see same program again.
                    self.early_stop_counter += 1
                    print(
                        f"[UNCOND] Detected Early Stop Again ({self.early_stop_counter}/{self.early_stop_threshold})"
                    )
                else:
                    print(
                        "[UNCOND] Previous Detection has ended - different unique programs"
                    )
                    self.early_stop_detected = False
                    self.early_stop_counter = 0
                    self.early_stop_detected_program = None
            else:
                return_in_last_line = "return" in unique_program.splitlines()[-1]
                if return_in_last_line:
                    self.early_stop_detected = True
                    self.early_stop_detected_program = unique_program
                    self.early_stop_counter = 1
                    print(
                        f"[UNCOND] Detected Early Stop ({self.early_stop_counter}/{self.early_stop_threshold})"
                    )
                    print("$" * 10)
                    print(self.early_stop_detected_program)
                    print("$" * 10)

        if self.early_stop_counter >= self.early_stop_threshold:
            print("[UNCOND] Perform Early Stop")
            self.perform_early_stop = True

        dynamic_signals = []
        for executable_partial_program_code in executable_partial_programs:
            for test_case, program_execution in self.program_executions[
                executable_partial_program_code
            ].items():
                trace = program_execution
                function_name, args_str, _ = parse_mbpp_assert_statement(test_case)
                innvocation = f"{function_name}{args_str}"
                dynamic_signal = MULTIPLE_CANDIDATES_DYNAMIC_SIGNAL_PATTERN.format(
                    function_code=executable_partial_program_code,
                    test_case=innvocation,
                    trace=trace,
                )
                dynamic_signals.append(dynamic_signal)

        dynamic_signal_text = ""
        if dynamic_signals:
            dynamic_signals = "\n".join(dynamic_signals)
            dynamic_signal_text = MULTIPLE_CANDIDATES_DYNAMIC_SIGNAL_PROMPT.format(
                dynamic_signals=dynamic_signals
            )

        # if the last line ends with : add a pass
        self.current_dynamic_signal[dynamic_signal_type] = dynamic_signal_text
        executable_partial_program_code = self._extract_partial_executions(new_code)
        self.current_debug_data[dynamic_signal_type] = (
            executable_partial_program_code,
            new_code,
        )
        return (
            self.current_dynamic_signal[dynamic_signal_type],
            self.current_debug_data[dynamic_signal_type],
        )

    def _extract_new_code(self, input_ids):
        crop_idx = self.initial_prompt_input_ids_len
        new_code, _ = extract_new_tokens(self.tokenizer, input_ids.clone(), crop_idx)
        if self.prompt_type == PROMPT_TYPE__DEEPSEEK_BASE:
            # Nothing to do, everything after [BEGIN] should be code
            pass
        if self.prompt_type in (
            PROMPT_TYPE__DEEPSEEK_INSTRUCT,
            PROMPT_TYPE__INSTRUCT_LONG_CODE_PROMPT,
        ):
            new_code = slice_prompt_after_markers(
                new_code, marker=INSTRUCT_MODEL_PYTHON_CODE_START
            )
        return new_code

    def _do_generate_new_signal(self, dynamic_signal_type, new_code):
        guidance_strategy = self.guidance_strategy
        if (guidance_strategy == GUIDANCE_STRATEGY__PERSISTENT_PREFIX_GUIDANCE) and (
            dynamic_signal_type != DYNAMIC_SIGNAL__MULTIPLE_CANDIDATES_EXECUTION
        ):
            guidance_strategy = GUIDANCE_STRATEGY__LINE_GUIDANCE

        generate_new_signal = False
        if guidance_strategy == GUIDANCE_STRATEGY__TOKEN_GUIDANCE:
            generate_new_signal = True
        if guidance_strategy == GUIDANCE_STRATEGY__LINE_GUIDANCE:
            lines_count = new_code.count("\n")
            if lines_count > self.lines_count:
                self.lines_count = lines_count
                generate_new_signal = True
        if guidance_strategy == GUIDANCE_STRATEGY__PERSISTENT_PREFIX_GUIDANCE:
            assert (
                dynamic_signal_type == DYNAMIC_SIGNAL__MULTIPLE_CANDIDATES_EXECUTION
            ), f"Unsupported Signal Type: {dynamic_signal_type}"
            # iterate over the new codes that were generated and check if new code is a prefix
            if not self.current_bs_candidates_count:
                generate_new_signal = True
            for idx, current_new_code_sample in enumerate(
                self.current_bs_candidates_count
            ):
                if not current_new_code_sample.startswith(new_code):
                    # print(f"New Code:\n{new_code}")
                    # print()
                    # print(f"#{idx} Sample:\n{current_new_code_sample}")
                    generate_new_signal = True
                    break

        if not self.current_dynamic_signal[dynamic_signal_type]:
            generate_new_signal = True
        return generate_new_signal

    def _extract_partial_execution_dynamic_signals(
        self, dynamic_signal_type, input_ids
    ):
        new_code = self._extract_new_code(input_ids)
        generate_new_signal = self._do_generate_new_signal(
            dynamic_signal_type, new_code
        )
        if not generate_new_signal:
            return (
                self.current_dynamic_signal[dynamic_signal_type],
                self.current_debug_data[dynamic_signal_type],
            )
        self.generate_new_signal = generate_new_signal

        executable_partial_program_code = self._extract_partial_executions(new_code)

        dynamic_signals = []
        if executable_partial_program_code:
            for test_case, program_execution in self.program_executions[
                executable_partial_program_code
            ].items():
                trace = program_execution
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
        self.current_dynamic_signal[dynamic_signal_type] = dynamic_signal_text
        self.current_debug_data[dynamic_signal_type] = debug_data
        return dynamic_signal_text, debug_data

    def unify_dynamic_signals(self, input_ids, dynamic_signals_text):
        new_code, new_code_tokens = extract_new_tokens(
            self.tokenizer, input_ids, self.initial_prompt_input_ids_len
        )

        unified_dynamic_signal_text = ""
        for dynamic_signal in self.dynamic_signals_types:
            unified_dynamic_signal_text += dynamic_signals_text[dynamic_signal]

        unified_dynamic_signal_prompt = self.initial_prompt
        if unified_dynamic_signal_text:
            if self.prompt_type in (
                PROMPT_TYPE__DEEPSEEK_INSTRUCT,
                PROMPT_TYPE__INSTRUCT_LONG_CODE_PROMPT,
            ):
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
        if self.use_local_hf_model and self.device:
            dynamic_signal_input_ids = torch.cat(
                [
                    unified_dynamic_signal_prompt_tokens["input_ids"].to(self.device),
                    new_code_tokens.clone(),
                ],
                dim=1,
            )
        elif self.use_inference_endpoint:
            dynamic_signal_input_ids = torch.cat(
                [
                    unified_dynamic_signal_prompt_tokens["input_ids"],
                    new_code_tokens.clone(),
                ],
                dim=1,
            )
        return dynamic_signal_input_ids

    def check_dynamic_early_stop_wrapper(self, dynamic_signal_input_ids):
        try:
            self.check_dynamic_early_stop(dynamic_signal_input_ids)
        except:
            print(
                "Exception occured durring early stop for dynamic signal, since its an optimization we reset and ignore"
            )
            self.dynamic_early_stop_detected = False
            self.dynamic_early_stop_detected_program = None
            self.dynamic_early_stop_counter = 0
            self.dynamic_early_stop_threshold = 4

    def check_dynamic_early_stop(self, dynamic_signal_input_ids):
        assert self.model_name
        new_codes = inference_endpoint_utils__sample_code_beam_search(
            dynamic_signal_input_ids,
            tokenizer=self.tokenizer,
            execution_manager=self.execution_manager,
            stats_manager=self.stats_manager,
            candidates_count=self.bs_candidates_count,
            temperature=self.temperature,
            bs_completion_horizon=self.bs_completion_horizon,
            model_name=self.model_name,
            prompt_with_cot=self.prompt_with_cot,
        )

        new_codes = list(set(new_codes))
        executable_partial_programs = []

        # print(f"New Codes: {len(new_codes)}")
        for idx, new_code in enumerate(new_codes):
            try:
                # print(f"#{idx + 1} Extracting Partial Executable")
                executable_partial_program_code = (
                    self.execution_manager.extract_partial_executable_program(new_code)
                )
            except ValueError:
                # print(f"#{idx + 1} Error Extracting Partial Executable")
                continue
            executable_partial_programs.append(executable_partial_program_code)
        executable_partial_programs = list(set(executable_partial_programs))

        # print(f"Executable Programs: {len(executable_partial_programs)}")
        if executable_partial_programs:
            self.current_bs_candidates_count = executable_partial_programs

        if self.dynamic_early_stop_detected and len(executable_partial_programs) != 1:
            print(
                f"[DYNAMIC] Cancel Early Stop since no unique executables ({self.dynamic_early_stop_counter}/{self.dynamic_early_stop_threshold})"
            )
            self.dynamic_early_stop_detected = False
            self.dynamic_early_stop_counter = 0
            self.dynamic_early_stop_detected_program = None

        if len(executable_partial_programs) == 1:
            unique_program = executable_partial_programs[0].strip()
            if self.dynamic_early_stop_detected:
                if self.dynamic_early_stop_detected_program == unique_program:
                    # We see same program again.
                    self.dynamic_early_stop_counter += 1
                    print(
                        f"[DYNAMIC] Detected Early Stop Again ({self.dynamic_early_stop_counter}/{self.dynamic_early_stop_threshold})"
                    )
                else:
                    print(
                        "[DYNAMIC] Previous Detection has ended - different unique programs"
                    )
                    self.dynamic_early_stop_detected = False
                    self.dynamic_early_stop_counter = 0
                    self.dynamic_early_stop_detected_program = None
            else:
                return_in_last_line = "return" in unique_program.splitlines()[-1]
                if return_in_last_line:
                    self.dynamic_early_stop_detected = True
                    self.dynamic_early_stop_detected_program = unique_program
                    self.dynamic_early_stop_counter = 1
                    print(
                        f"[DYNAMIC] Detected Early Stop ({self.dynamic_early_stop_counter}/{self.dynamic_early_stop_threshold})"
                    )
                    print("$" * 10)
                    print(self.dynamic_early_stop_detected_program)
                    print("$" * 10)

        if self.dynamic_early_stop_counter >= self.dynamic_early_stop_threshold:
            print("[DYNAMIC] Perform Early Stop")
            self.perform_dynamic_early_stop = True

    def extract_dynamic_signal_input_ids(self, input_ids):
        dynamic_signals_text = {}
        debug_data = {}

        for dynamic_signal_type in self.dynamic_signals_types:
            (
                dynamic_signals_text[dynamic_signal_type],
                debug_data[dynamic_signal_type],
            ) = self.dynamic_signal_handlers()[dynamic_signal_type](
                self, dynamic_signal_type, input_ids
            )

        dynamic_signal_input_ids = self.unify_dynamic_signals(
            input_ids, dynamic_signals_text
        )
        if self.generate_new_signal and self.early_stop_detected:
            self.check_dynamic_early_stop_wrapper(dynamic_signal_input_ids)
        debug_data = debug_data.get(
            DYNAMIC_SIGNAL__MULTIPLE_CANDIDATES_EXECUTION, ("", "")
        )

        return dynamic_signal_input_ids, debug_data

    def _extract_partial_executions(self, new_code) -> str:
        executable_partial_program_code = ""
        try:
            executable_partial_program_code = (
                self.execution_manager.extract_partial_executable_program(new_code)
            )
            if (
                executable_partial_program_code
                and executable_partial_program_code not in self.program_executions
            ):
                print(f"Executing:\n {executable_partial_program_code}\n")
                self.program_executions[executable_partial_program_code] = (
                    self.execution_manager.execute_test_cases(
                        executable_partial_program_code, self.test_cases
                    )
                )
        except ValueError:
            executable_partial_program_code = ""
            pass

        return executable_partial_program_code

    def check_early_termination(self):
        if self.task_id is None:
            return
        global_cache_solved_task_id_path = os.path.join(
            self.solved_tasks_cache_dir, f"{self.task_id}"
        )
        if os.path.exists(global_cache_solved_task_id_path):
            print(f"Task {self.task_id} already solved, early termination requested")
            raise EarlyTerminationExceptin(
                f"Task {self.task_id} already solved, early termination requested"
            )
