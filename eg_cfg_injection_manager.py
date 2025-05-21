import torch
import re
from code_generation_adapter import CodeGenerationAdapter
from probs_utils import apply_guidance, mask_topk_probs_log_stable
from model_utils import extract_new_tokens
from consts import *

TASKS_ADAPTERS = {TASK__CODE_GENERATION: CodeGenerationAdapter}


class EgCfgDetector:
    def __init__(self, tokenizer, initial_prompt_input_ids_len):
        self.tokenizer = tokenizer
        self.initial_prompt_input_ids_len = initial_prompt_input_ids_len
        self.eg_cfg_enabled = False
        self.start_detected = 0
        self.end_detected = 0
        self.eg_cfg_count = 0

    def is_eg_cfg_enabled(self, input_ids):
        if not self.eg_cfg_enabled:
            if self.detect_start(input_ids):
                self.start_detected += 1
                self.eg_cfg_enabled = True
        else:
            if self.detect_end(input_ids):
                self.end_detected += 1
                self.eg_cfg_enabled = False
        return self.eg_cfg_enabled

    def detect_start(self, input_ids):
        raise NotImplementedError()

    def detect_end(self, input_ids):
        raise NotImplementedError()


class FunctinoSigEgCfgDetector(EgCfgDetector):
    def __init__(
        self, tokenizer, initial_prompt_input_ids_len, function_name, end_string
    ):
        super().__init__(tokenizer, initial_prompt_input_ids_len)
        self.function_name = function_name
        self.end_string = end_string
        self.function_start_idx = None

    def detect_start(self, input_ids):
        new_text, new_text_tokens = extract_new_tokens(
            self.tokenizer, input_ids, self.initial_prompt_input_ids_len
        )
        new_text_lines = new_text.split("\n")
        last_line = new_text_lines[-1]

        pattern = rf"^def\s+{self.function_name}\s*\("
        result = bool(re.search(pattern, last_line))
        if result:
            # Backward iterate through input_ids[0]
            for tok_idx in reversed(range(input_ids.shape[1])):
                token_id = input_ids[0, tok_idx].item()  # get int value
                tok = self.tokenizer.decode([token_id])
                if tok.startswith("def"):
                    if self.function_start_idx is None:
                        self.function_start_idx = tok_idx
                    break
        return result

    def detect_end(self, input_ids):
        new_text, new_text_tokens = extract_new_tokens(
            self.tokenizer, input_ids, self.initial_prompt_input_ids_len
        )
        new_text_lines = new_text.split("\n")
        last_line = new_text_lines[-1]
        result = last_line.startswith(self.end_string)
        return result


TASKS_DETECDORS = {TASK__CODE_GENERATION: FunctinoSigEgCfgDetector}


class EgCfgInjectionManager:
    def __init__(
        self,
        tokenizer,
        task,
        task_kwargs,
        gamma,
        detector_kwargs,
        use_detector=False,
        top_probs_count=0,
        debug_mode=False,
    ):
        self.tokenizer = tokenizer
        self.gamma = gamma
        assert task in TASKS_ADAPTERS
        self.adapter = TASKS_ADAPTERS[task](**task_kwargs)
        if use_detector:
            self.detector = TASKS_DETECDORS[task](**detector_kwargs)
            self.adapter.detector = self.detector
        self.top_probs_count = top_probs_count
        self.debug_mode = debug_mode

    def is_eg_cfg_enabled(self, input_ids):
        if self.gamma == 0:
            return False
        return self.detector.is_eg_cfg_enabled(input_ids)

    def is_top_probs_enabled(self):
        return self.top_probs_count > 0

    def mask_top_probs(self, probs: torch.Tensor):
        assert self.is_top_probs_enabled()
        return mask_topk_probs_log_stable(probs, self.top_probs_count)

    def extract_dynamic_signal_input_ids(self, input_ids):
        return self.adapter.extract_dynamic_signal_input_ids(input_ids)

    def apply_guidance(self, P, P_c, eps=1e-8, debug=False):
        return apply_guidance(
            P, P_c, self.gamma, eps=eps, tokenizer=self.tokenizer, debug=debug
        )

    def early_stop_detected(self):
        return self.adapter.query_early_stop()
