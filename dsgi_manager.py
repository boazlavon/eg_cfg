import torch
import black
import re

from collections import OrderedDict
from mbpp_utils import parse_mbpp_assert_statement
from code_generation import remove_comments_and_docstrings, is_valid_python
from execution_manager import ExecutionManager

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
        self, prompt, function_signature, test_cases, tokenizer, device, gamma=0.3
    ):
        self.tokenizer = tokenizer
        self.function_signature = function_signature
        self.test_cases = test_cases
        self.program_executions = OrderedDict()
        self.dynamic_signals_prompts = OrderedDict()
        self.execution_manager = ExecutionManager()

        prompt_token_ids = tokenizer(prompt, return_tensors="pt")
        self.prompt = prompt
        self.prompt_input_ids = prompt_token_ids["input_ids"]  # shape: (1, prompt_len)
        self.prompt_input_ids_len = self.prompt_input_ids.shape[1]
        self.device = device
        self.gamma = gamma

    def extract_new_tokens(self, input_ids: torch.Tensor) -> str:
        if input_ids.dim() != 2 or input_ids.size(0) != 1:
            raise ValueError("Expected input_ids to have shape (1, sequence_length)")

        new_token_ids = input_ids[:, self.prompt_input_ids_len :]
        new_text = self.tokenizer.batch_decode(new_token_ids, skip_special_tokens=True)[
            0
        ]
        return new_text, new_token_ids

    def make_executable(
        self, partial_program_code: str, fallback_to_prompt: bool = True
    ) -> str:
        function_signature = self.function_signature
        lines = partial_program_code.split("\n")
        executable_code = ""

        while lines:
            executable_code = "\n".join(lines)
            if is_valid_python(executable_code) and executable_code.startswith(
                function_signature
            ):
                break

            # Remove last line and try again
            last_line = lines.pop()
            if not lines:
                break  # Stop if there are no lines left

            executable_code = "\n".join(lines)
            if is_valid_python(executable_code) and executable_code.startswith(
                function_signature
            ):
                break

            # If removing doesn't work, replace last line with 'pass' (preserving indentation)
            indent = re.match(r"\s*", last_line).group(0)
            lines.append(f"{indent}pass")
            executable_code = "\n".join(lines)
            if is_valid_python(executable_code) and executable_code.startswith(
                function_signature
            ):
                break
            lines.pop()  # Remove the pass if it's still invalid

        if (
            not is_valid_python(executable_code)
            or not executable_code.startswith(function_signature)
        ) and fallback_to_prompt:
            prompt_lines = function_signature.split("\n")
            last_line = prompt_lines[-1]
            indent = re.match(r"\s*", last_line).group(0)
            if not indent:
                indent = "   "
            executable_code = f"{function_signature}\n{indent}pass"
            executable_code = black.format_str(
                executable_code, mode=black.FileMode(line_length=1024)
            )

        return executable_code

    def extract_partial_program(self, input_ids: torch.Tensor) -> str:
        new_text, _ = self.extract_new_tokens(input_ids)
        partial_program_code = f"{self.function_signature}\n{new_text}"
        executable_partial_program_code = self.make_executable(partial_program_code)
        return partial_program_code, executable_partial_program_code, new_text

    def extract_partial_executions(self, input_ids: torch.Tensor) -> str:
        _, executable_partial_program_code, new_text = self.extract_partial_program(
            input_ids.clone()
        )
        executable_partial_program_code = remove_comments_and_docstrings(
            executable_partial_program_code
        )

        if executable_partial_program_code not in self.program_executions:
            self.program_executions[executable_partial_program_code] = {}
            self.dynamic_signals_prompts[executable_partial_program_code] = {}
            for test_case in self.test_cases:
                try:
                    function_name, args_str, expected_result_str = (
                        parse_mbpp_assert_statement(test_case)
                    )
                    innvocation = f"{function_name}{args_str}"
                    test_case_code = f"{executable_partial_program_code}\n{innvocation}"
                    assert is_valid_python(
                        test_case_code
                    ), f"Invalid Test Case: {test_case}"
                    test_case_code = black.format_str(
                        test_case_code, mode=black.FileMode(line_length=1024)
                    )
                    assert is_valid_python(
                        test_case_code
                    ), f"Invalid Test Case: {test_case}"
                except:
                    continue

                try:
                    program_execution, _, _ = self.execution_manager.execute(
                        test_case_code
                    )
                except:
                    print("Problem on program execution")
                    continue

                self.program_executions[executable_partial_program_code][
                    test_case
                ] = program_execution
                # self.dynamic_signals_prompts[executable_partial_program_code][
                #     test_case
                # ] = self.to_dynamic_signal_prompt(executable_partial_program_code, test_case, program_execution)
        self.dynamic_signals_prompts[executable_partial_program_code] = (
            self.to_dynamic_signal_prompt(
                self.program_executions[executable_partial_program_code]
            )
        )
        print(self.dynamic_signals_prompts[executable_partial_program_code])
        return (executable_partial_program_code, new_text)

    def to_dynamic_signal_prompt(self, executions):
        dynamic_signals = []
        for test_case, program_execution in executions.items():
            trace = program_execution.to_compact_json()
            function_name, args_str, expected_result_str = parse_mbpp_assert_statement(
                test_case
            )
            innvocation = f"{function_name}{args_str}"
            dynamic_signal = SINGLE_DYNAMIC_SIGNAL_PATTERN.format(
                test_case=innvocation, trace=trace
            )
            dynamic_signals.append(dynamic_signal)
        dynamic_signals = "\n".join(dynamic_signals)
        dynamic_signal_prompt = DYNAMIC_SIGNAL_PROMPT.format(
            dynamic_signals=dynamic_signals
        )
        full_dynamic_signal_prompt = self.prompt.replace(
            DYNAMIC_SIGNAL_PROMPT_REPLACE_STRING, dynamic_signal_prompt
        )
        full_dynamic_signal_prompt = full_dynamic_signal_prompt.replace(
            "Response:\n\n", "Response:\n"
        )
        return full_dynamic_signal_prompt

    def to_dynamic_signal_input_ids(self, input_ids):
        executable_partial_program_code, new_text = self.extract_partial_executions(
            input_ids
        )
        _, new_tokens = self.extract_new_tokens(input_ids)
        dynamic_signals_input_ids = {}
        full_dynamic_signal_prompt = self.dynamic_signals_prompts[
            executable_partial_program_code
        ]
        full_dynamic_signal_prompt_tokens = self.tokenizer(
            full_dynamic_signal_prompt, return_tensors="pt"
        )
        combined_input_ids = torch.cat(
            [
                full_dynamic_signal_prompt_tokens["input_ids"].to(self.device),
                new_tokens.clone(),
            ],
            dim=1,
        )
        dynamic_signals_input_ids = {None: combined_input_ids}
        return dynamic_signals_input_ids, executable_partial_program_code, new_text

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

    def apply_guidance_from_probs(self, p, p_gs, eps=1e-8):
        R = torch.ones_like(p)
        for p_g in p_gs:
            R *= (p_g + eps) / (p + eps)
        # R = (p_g + eps) / (p + eps)

        p_guided = p * R**self.gamma
        p_guided = p_guided / p_guided.sum(
            dim=-1, keepdim=True
        )  # normalize across vocab
        return p_guided
