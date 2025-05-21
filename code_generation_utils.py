import torch
import re
import ast
import tokenize
from io import StringIO
from transformers import StoppingCriteria
from transformers import StoppingCriteriaList
from inference_endpoint_utils import extract_python_code
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


def has_function_definition(code, function_name):
    pattern = rf"def\s+.*{re.escape(function_name)}\s*\("
    return re.search(pattern, code) is not None


class CodeGenStopCriteria(StoppingCriteria):
    def __init__(
        self,
        tokenizer,
        bs_completion_horizon=None,
        function_name=None,
        is_instruct=True,
    ):
        self.tokenizer = tokenizer
        self.bs_completion_horizon = bs_completion_horizon
        self.function_name = function_name
        self.is_instruct = is_instruct
        self.code_started = False
        self.code_ended = False

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

    def check_stop_instruct(
        self, token: str, count_lines: bool = True
    ) -> tuple[bool, bool]:
        should_stop = False
        discard_token = False

        self.generated_text += token
        if token == "<|endoftext|>":
            should_stop = True

        if not self.code_started and "```" in token:
            self.code_started = True
            return should_stop, discard_token

        if self.code_started and not self.code_ended and "```" in token:
            self.code_ended = True
            should_stop = True
            discard_token = True
            return should_stop, discard_token

        def_count = self.generated_text.count("def")
        if (
            self.code_started
            and not self.def_reached
            and self.function_name is not None
        ):
            self.def_reached = has_function_definition(
                self.generated_text, self.function_name
            )

        elif self.def_reached and "\n" in self.previous_token:
            if (
                token.startswith(" ")
                or token.startswith("\t")
                or token.startswith("\n")
            ):
                if not self.inside_function_body:
                    self.inside_function_body = True
                    if count_lines:
                        self.function_body_line_count = 1
                else:
                    if count_lines:
                        self.function_body_line_count += 1

                if (
                    count_lines
                    and self.bs_completion_horizon is not None
                    and self.function_body_line_count >= self.bs_completion_horizon
                ):
                    should_stop = True
                    discard_token = True

            elif self.inside_function_body:
                pass
                # We've exited the function body
                # should_stop = True
                # discard_token = True

        self.previous_token = token
        return should_stop, discard_token

    def check_stop(self, token: str, count_lines: bool = True) -> tuple[bool, bool]:
        should_stop = False
        discard_token = False

        self.generated_text += token
        if token == "<|endoftext|>":
            should_stop = True

        if not self.def_reached and "def" in token:
            self.def_reached = True

        if self.def_reached and "\n" in self.previous_token:
            # stripped = token.lstrip("\n")
            # if stripped.startswith(" ") or stripped.startswith("\t"):
            if (
                token.startswith(" ")
                or token.startswith("\t")
                or token.startswith("\n")
            ):
                if not self.inside_function_body:
                    self.inside_function_body = True
                    if count_lines:
                        self.function_body_line_count = 1
                else:
                    if count_lines:
                        self.function_body_line_count += 1

                if (
                    count_lines
                    and self.bs_completion_horizon is not None
                    and self.function_body_line_count >= self.bs_completion_horizon
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
        if self.is_instruct:
            should_stop, discard_token = self.check_stop_instruct(token_str)
        else:
            should_stop, discard_token = self.check_stop(token_str)

        self.discard_last_token = discard_token
        return should_stop


def raw_outputs_to_new_code(
    outputs,
    tokenizer,
    initial_prompt_input_ids_len,
    prompt_type=None,
    validate=True,
    stats_manager=None,
):
    new_codes = []
    for output in outputs:
        try:
            output = output.unsqueeze(0)
            output_text, output_tokens = extract_new_tokens(
                tokenizer, output, initial_prompt_input_ids_len
            )
            if stats_manager is not None:
                stats_manager.increate_counter("output_tokens", output_tokens.shape[1])
            if prompt_type == PROMPT_TYPE__DEEPSEEK_BASE:
                output_text = output_text.replace(
                    DYNAMIC_SIGNAL_PROMPT_REPLACE_STRING_BASE_END, ""
                ).strip()
            if prompt_type in (
                PROMPT_TYPE__DEEPSEEK_INSTRUCT,
                PROMPT_TYPE__INSTRUCT_LONG_CODE_PROMPT,
            ):
                extracted_code = extract_python_code(output_text)
                if not extracted_code:
                    extracted_code = output_text.split(
                        INSTRUCT_MODEL_PYTHON_CODE_START, 1
                    )[1].rstrip()
                    if "```" in extracted_code:
                        extracted_code = extracted_code.split("```")[0]
            if validate:
                assert is_valid_python(extracted_code)
                # new_code = black.format_str(
                #     new_code, mode=black.FileMode(line_length=1024)
                # )
        except:
            continue
        new_codes.append(extracted_code)
    assert new_codes
    return new_codes


def slice_prompt_after_markers(text: str, marker: str, marker2: str = None) -> str:
    result = ""

    if marker in text:
        result = text.rsplit(marker, 1)[-1]
        if not result.strip():
            result = ""
    if marker2 is not None and result:
        if marker2 in result:
            result = result.rsplit(marker2, 1)[-1]
            if not result.strip():
                result = ""
        else:
            result = ""

    return result


def prime_stopping_criteria(
    tokenizer, inputs, stop_criteria_list, marker, marker2=None
):
    # Decode full prompt text
    prompt_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)

    # Slice from the last [BEGIN] marker
    prompt_text = slice_prompt_after_markers(prompt_text, marker, marker2)
    if not prompt_text:
        return

    # Tokenize and decode each token
    prompt_tokens = tokenizer.tokenize(prompt_text)
    decoded_tokens = [tokenizer.convert_tokens_to_string([t]) for t in prompt_tokens]

    # Feed each token into all stopping criteria with count_lines=False
    for criteria in stop_criteria_list:
        for token in decoded_tokens:
            criteria.check_stop_instruct(token, count_lines=False)


def generate_code_solutions(
    model,
    tokenizer,
    eg_cfg_injection_manager,
    inputs,
    max_new_tokens=MAX_NEW_TOKENS,
    num_return_sequences=1,
    temperature=1,
    bs_completion_horizon=None,
    function_name=None,
    do_sample=False,
    prompt_type=None,
    stats_manager=None,
):
    if stats_manager is not None:
        stats_manager.increate_counter("input_tokens", inputs["input_ids"].shape[1])
    stop_criteria_list = [
        CodeGenStopCriteria(
            tokenizer,
            bs_completion_horizon=bs_completion_horizon,
            function_name=function_name,
        )
        for _ in range(num_return_sequences)
    ]
    stopping_criteria = StoppingCriteriaList(stop_criteria_list)
    if prompt_type == PROMPT_TYPE__DEEPSEEK_BASE:
        prime_stopping_criteria(
            tokenizer,
            inputs,
            stop_criteria_list,
            marker=DYNAMIC_SIGNAL_PROMPT_BASE_MODEL_START_FUNCTION_MARKER,
        )
        eos_token_id = tokenizer.eos_token_id
        pad_token_id = tokenizer.eos_token_id
    if prompt_type in (
        PROMPT_TYPE__DEEPSEEK_INSTRUCT,
        PROMPT_TYPE__INSTRUCT_LONG_CODE_PROMPT,
    ):
        prime_stopping_criteria(
            tokenizer,
            inputs,
            stop_criteria_list,
            marker=DYNAMIC_SIGNAL_PROMPT_INSTRUCT_MODEL_START_FUNCTION_MARKER,
            marker2=INSTRUCT_MODEL_PYTHON_CODE_START,
        )
        stop_id = tokenizer.convert_tokens_to_ids("<|EOT|>")
        assert isinstance(stop_id, int), "Invalid tokenizer, EOT id not found"
        eos_token_id = stop_id
        pad_token_id = stop_id

    if do_sample:
        sampling_kwargs = {
            "do_sample": True,
            "num_return_sequences": num_return_sequences,
            "top_p": 0.95,
            "temperature": temperature,
        }
    else:
        sampling_kwargs = {
            "do_sample": False,
            # "num_return_sequences": 1,
        }

    model.eval()
    with torch.inference_mode():
        # transformers/generation/utils.py
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            num_beams=1,
            stopping_criteria=stopping_criteria,
            eg_cfg_injection_manager=eg_cfg_injection_manager,
            use_cache=True,
            **sampling_kwargs,
        )
    processed_outputs = []
    for i, output_ids in enumerate(outputs):
        criteria = stop_criteria_list[i]
        if criteria.discard_last_token:
            output_ids = output_ids[:-1]
        processed_outputs.append(output_ids)
    return processed_outputs
