import torch
import black
import ast
import re


class CodeGenStopCriteria(torch.utils.data.Dataset):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.generated_text = ""
        self.previous_newline_index = -1
        self.remove_last_newline = False

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        new_token = self.tokenizer.decode(input_ids[0][-1])
        if "<|end_of_text|>" in new_token:
            new_token = new_token.replace("<|end_of_text|>", "")
            self.generated_text += new_token
            return True

        self.generated_text += new_token
        if "\n" in new_token:
            if self.previous_newline_index != -1:
                current_newline_index = len(self.generated_text) - 1
                substring = self.generated_text[
                    self.previous_newline_index + 1 : current_newline_index
                ]

                if substring.startswith(" ") or substring.startswith("\t"):
                    self.previous_newline_index = current_newline_index
                    return False
                else:
                    # print(self.previous_newline_index, current_newline_index, substring)
                    self.remove_last_newline = True
                    self.generated_text = self.generated_text[
                        : self.previous_newline_index
                    ]
                    self.generated_text = self.generated_text.replace("\n\n", "\n")
                    return True
            else:
                self.previous_newline_index = len(self.generated_text) - 1
            return False

        return False


class DsgiManager:
    def __init__(self, prompt, function_signature, tokenizer):
        self.tokenizer = tokenizer
        self.function_signature = function_signature

        prompt_token_ids = tokenizer(prompt, return_tensors="pt")
        self.prompt_input_ids = prompt_token_ids["input_ids"]  # shape: (1, prompt_len)
        self.prompt_input_ids_len = self.prompt_input_ids.shape[1]

    def extract_new_tokens(self, input_ids: torch.Tensor) -> str:
        if input_ids.dim() != 2 or input_ids.size(0) != 1:
            raise ValueError("Expected input_ids to have shape (1, sequence_length)")

        new_token_ids = input_ids[:, self.prompt_input_ids_len :]
        new_text = self.tokenizer.batch_decode(new_token_ids, skip_special_tokens=True)[
            0
        ]
        return new_text

    def extract_partial_program(self, input_ids: torch.Tensor) -> str:
        new_text = self.extract_new_tokens(input_ids)
        partial_program_code = f"{self.function_signature}\n{new_text}"
        executable_partial_program_code = self.make_executable(partial_program_code)
        return partial_program_code, executable_partial_program_code

    def make_executable(
        self, partial_program_code: str, fallback_to_prompt: bool = True
    ) -> str:
        function_signature = self.function_signature
        lines = partial_program_code.split("\n")
        executable_code = ""

        while lines:
            executable_code = "\n".join(lines)
            if self.is_valid_python(executable_code) and executable_code.startswith(
                function_signature
            ):
                break

            # Remove last line and try again
            last_line = lines.pop()
            if not lines:
                break  # Stop if there are no lines left

            executable_code = "\n".join(lines)
            if self.is_valid_python(executable_code) and executable_code.startswith(
                function_signature
            ):
                break

            # If removing doesn't work, replace last line with 'pass' (preserving indentation)
            indent = re.match(r"\s*", last_line).group(0)
            lines.append(f"{indent}pass")
            executable_code = "\n".join(lines)
            if self.is_valid_python(executable_code) and executable_code.startswith(
                function_signature
            ):
                break
            lines.pop()  # Remove the pass if it's still invalid

        if (
            not self.is_valid_python(executable_code)
            or not executable_code.startswith(function_signature)
        ) and fallback_to_prompt:
            prompt_lines = function_signature.split("\n")
            last_line = prompt_lines[-1]
            indent = re.match(r"\s*", last_line).group(0)
            if not indent:
                indent = "   "
            executable_code = f"{function_signature}\n{indent}pass"
            executable_code = black.format_str(executable_code, mode=black.FileMode())

        return executable_code

    def is_valid_python(self, code: str) -> bool:
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False


def generate_solution(prompt, function_signature, model, tokenizer, device):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    stop_criteria = CodeGenStopCriteria(tokenizer)

    with torch.no_grad():
        # /a/home/cc/students/cs/boazlavon/miniconda3/envs/trepan-xpy-env/lib/python3.9/site-packages/transformers/generation/utils.py
        import ipdb

        ipdb.set_trace()
        dsgi_manager = DsgiManager(prompt, function_signature, tokenizer)
        _ = model.generate(
            **inputs,
            max_new_tokens=512,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            stopping_criteria=[stop_criteria],
            dsgi_manager=dsgi_manager,
        )

    # _ = tokenizer.decode(outputs[0], skip_special_tokens=True)
    new_tokens = stop_criteria.generated_text
    return new_tokens
