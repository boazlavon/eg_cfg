import torch
import black
import re

from collections import OrderedDict
from execution_manager import ExecutionManager
from model_utils import extract_new_tokens
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


class DsgiManager:
    def __init__(
        self,
        initial_prompt,
        function_signature,
        test_cases,
        tokenizer,
        device,
        gamma=0.3,
    ):
        self.tokenizer = tokenizer
        self.function_signature = function_signature
        self.test_cases = test_cases
        self.program_executions = OrderedDict()
        self.execution_manager = ExecutionManager(tokenizer, function_signature)

        self.initial_prompt = initial_prompt
        prompt_token_ids = tokenizer(self.initial_prompt, return_tensors="pt")
        self.initial_prompt_input_ids = prompt_token_ids[
            "input_ids"
        ]  # shape: (1, prompt_len)
        self.initial_prompt_input_ids_len = self.initial_prompt_input_ids.shape[1]

        self.device = device
        self.gamma = gamma

    def extract_partial_executions(self, input_ids: torch.Tensor) -> str:
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

    def to_dynamic_signal_prompt(self, executions):
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

    def create_dynamic_signal_input_ids(
        self, executable_partial_program_code, new_code_tokens
    ):
        unified_dynamic_signal_prompt = self.to_dynamic_signal_prompt(
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

        dynamic_signals_input_ids = {"unified": dynamic_signal_input_ids}
        return dynamic_signals_input_ids

    def to_dynamic_signal_input_ids(self, input_ids):
        executable_partial_program_code = self.extract_partial_executions(input_ids)
        new_code, new_code_tokens = extract_new_tokens(
            self.tokenizer, input_ids, self.initial_prompt_input_ids_len
        )
        dynamic_signals_input_ids = self.create_dynamic_signal_input_ids(
            executable_partial_program_code, new_code_tokens
        )

        debug_data = executable_partial_program_code, new_code
        return dynamic_signals_input_ids, debug_data

    def print_top_k_token_probs(self, p1, p2, k=5):
        # Get top-k from p1 and p2
        top1_vals, top1_idxs = torch.topk(p1, k)
        top2_vals, top2_idxs = torch.topk(p2, k)

        def format_token(idx):
            return self.tokenizer.decode(idx).strip() or repr(
                self.tokenizer.decode(idx)
            )

        print(f"\nTop-{k} tokens in orignal_p and guided_p :")
        print(
            f"{'Token (orignal_p)':<15} {'orignal_p':>10}    {'Token (guided_p)':<15} {'guided_p':>10}"
        )
        print("-" * 56)

        for (idx1, val1), (idx2, val2) in zip(
            zip(top1_idxs[0], top1_vals[0]), zip(top2_idxs[0], top2_vals[0])
        ):
            token1 = format_token(idx1)
            token2 = format_token(idx2)
            print(
                f"{token1:<15} {val1.item():10.4f}    {token2:<15} {val2.item():10.4f}"
            )

    # Stay on topic with Classifier-Free Guidance
    # https://arxiv.org/abs/2306.17806
    # Inputs are the prior probability P and conditional probabilities P_cs
    # each associated with a different dynamic signal (c)
    def apply_guidance_from_probs(self, P, P_cs, eps=1e-8):
        R = torch.ones_like(P)
        for P_c in P_cs:
            R *= (P_c + eps) / (P + eps)

        P_guided = P * R**self.gamma
        P_guided = P_guided / P_guided.sum(
            dim=-1, keepdim=True
        )  # normalize across vocab
        return P_guided
