import torch
from code_generation_adapter import CodeGenerationAdapter
from probs_utils import apply_guidance

TASK__CODE_GENERATION = "CodeGeneration"
SUPPORTED_TASKS = (TASK__CODE_GENERATION,)

TASKS_ADAPTERS = {TASK__CODE_GENERATION: CodeGenerationAdapter}


class DsgiManager:
    def __init__(self, tokenizer, task, task_kwargs, gamma):
        self.tokenizer = tokenizer
        self.gamma = gamma
        assert task in TASKS_ADAPTERS
        self.adapter = TASKS_ADAPTERS[task](**task_kwargs)

    def extract_dynamic_signal_input_ids(self, input_ids):
        return self.adapter.extract_dynamic_signal_input_ids(input_ids)

    def apply_guidance(self, P, P_c, eps=1e-8, debug=False):
        return apply_guidance(
            P, P_c, self.gamma, eps=eps, tokenizer=self.tokenizer, debug=debug
        )
