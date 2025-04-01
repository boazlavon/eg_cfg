import torch
import ast
import black
import tokenize
from io import StringIO
from transformers import StoppingCriteria
from typing import Optional
from transformers import StoppingCriteriaList

from model_utils import extract_new_tokens
from consts import *


class DocstringRemover(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Str)
        ):
            node.body = node.body[1:]
        return node

    def visit_ClassDef(self, node):
        self.generic_visit(node)
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Str)
        ):
            node.body = node.body[1:]
        return node

    def visit_Module(self, node):
        self.generic_visit(node)
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Str)
        ):
            node.body = node.body[1:]
        return node


def remove_comments_and_docstrings(source: str) -> str:
    # Step 1: Remove comments using tokenize
    tokens = tokenize.generate_tokens(StringIO(source).readline)
    filtered_tokens = [
        (toknum, tokval)
        for toknum, tokval, _, _, _ in tokens
        if toknum != tokenize.COMMENT
    ]
    code_no_comments = tokenize.untokenize(filtered_tokens)

    # Step 2: Parse code and remove docstrings using AST
    try:
        tree = ast.parse(code_no_comments)
        tree = DocstringRemover().visit(tree)
        ast.fix_missing_locations(tree)
        code_no_docstrings = ast.unparse(tree)  # Python 3.9+
        return code_no_docstrings
    except Exception as e:
        # Fallback if parsing fails
        return code_no_comments


def is_valid_python(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


class CodeGenStopCriteria(StoppingCriteria):
    def __init__(self, tokenizer, max_function_body_lines=None):
        self.tokenizer = tokenizer
        self.max_function_body_lines = max_function_body_lines

        self.def_reached = False
        self.inside_function_body = False
        self.function_body_line_count = 0
        self.previous_token = ""
        self.stopped = False
        self.discard_last_token = False
        self.generated_text = ""

    def reset(self):
        self.def_reached = False
        self.inside_function_body = False
        self.function_body_line_count = 0
        self.previous_token = ""
        self.stopped = False
        self.discard_last_token = False

    def check_stop(self, token: str, count_lines: bool = True) -> tuple[bool, bool]:
        should_stop = False
        discard_token = False

        self.generated_text += token
        if token == "<|endoftext|>":
            should_stop = True

        if not self.def_reached and "def" in token:
            self.def_reached = True

        if self.def_reached and "\n" in self.previous_token:
            stripped = token.lstrip("\n")

            if stripped.startswith(" ") or stripped.startswith("\t"):
                if not self.inside_function_body:
                    self.inside_function_body = True
                    if count_lines:
                        self.function_body_line_count = 1
                else:
                    if count_lines:
                        self.function_body_line_count += 1

                if (
                    count_lines
                    and self.max_function_body_lines is not None
                    and self.function_body_line_count >= self.max_function_body_lines
                ):
                    should_stop = True
                    discard_token = True

            elif self.inside_function_body:
                # We've exited the function body
                should_stop = True
                discard_token = True

        self.previous_token = token
        return should_stop, discard_token

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        last_token_id = input_ids[0, -1]
        token_str = self.tokenizer.decode([last_token_id])
        should_stop, discard_token = self.check_stop(token_str)

        self.discard_last_token = discard_token
        return should_stop


def raw_outputs_to_new_code(
    outputs, tokenizer, initial_prompt_input_ids_len, prompt_type=None, validate=True
):
    new_codes = []
    for output in outputs:
        try:
            output = output.unsqueeze(0)
            new_code, _ = extract_new_tokens(
                tokenizer, output, initial_prompt_input_ids_len
            )
            if prompt_type == PROMPT_TYPE__DEEPSEEK_BASE:
                new_code = new_code.replace("[DONE]", "").strip()
            if validate:
                assert is_valid_python(new_code)
                new_code = black.format_str(
                    new_code, mode=black.FileMode(line_length=1024)
                )
        except:
            continue
        new_codes.append(new_code)
    return new_codes


def prime_stopping_criteria(tokenizer, inputs, stop_criteria_list, marker="[BEGIN]"):
    # Decode full prompt text
    prompt_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)

    # Slice from the last [BEGIN] marker
    if marker in prompt_text:
        prompt_text = prompt_text.rsplit(marker, 1)[-1]

        # Tokenize and decode each token
        prompt_tokens = tokenizer.tokenize(prompt_text)
        decoded_tokens = [
            tokenizer.convert_tokens_to_string([t]) for t in prompt_tokens
        ]

        # Feed each token into all stopping criteria with count_lines=False
        for criteria in stop_criteria_list:
            for token in decoded_tokens:
                criteria.check_stop(token, count_lines=False)


def generate_code_solutions(
    model,
    tokenizer,
    device,
    prompt,
    dsgi_manager,
    max_new_tokens=MAX_NEW_TOKENS,
    num_return_sequences=1,
    inputs=None,
    max_function_body_lines=None,
    do_sample=False,
    prompt_type=None,
):
    if inputs is None:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

    stop_criteria_list = [
        CodeGenStopCriteria(tokenizer, max_function_body_lines=max_function_body_lines)
        for _ in range(num_return_sequences)
    ]
    stopping_criteria = StoppingCriteriaList(stop_criteria_list)
    assert prompt_type == PROMPT_TYPE__DEEPSEEK_BASE
    if prompt_type == PROMPT_TYPE__DEEPSEEK_BASE:
        prime_stopping_criteria(
            tokenizer,
            inputs,
            stop_criteria_list,
            marker=DYNAMIC_SIGNAL_PROMPT_REPLACE_STRING_BEGIN,
        )

    if do_sample:
        sampling_kwargs = {
            "do_sample": True,
            "num_return_sequences": num_return_sequences,
            "top_p": 0.95,
            "temperature": 0.8,
        }
    else:
        sampling_kwargs = {
            "do_sample": False,
            "num_return_sequences": 1,
        }

    with torch.no_grad():
        # /a/home/cc/students/cs/boazlavon/miniconda3/envs/trepan-xpy-env/lib/python3.9/site-packages/transformers/generation/utils.py
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            num_beams=1,
            stopping_criteria=stopping_criteria,
            dsgi_manager=dsgi_manager,
            **sampling_kwargs
        )

    processed_outputs = []
    for i, output_ids in enumerate(outputs):
        criteria = stop_criteria_list[i]
        if criteria.discard_last_token:
            output_ids = output_ids[:-1]
        processed_outputs.append(output_ids)
    return processed_outputs
