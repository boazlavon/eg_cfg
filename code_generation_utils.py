import torch
import ast
import tokenize
from io import StringIO


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


class CodeGenStopCriteria(torch.utils.data.Dataset):
    def __init__(self, tokenizer, max_lines=None):
        self.tokenizer = tokenizer
        self.generated_text = ""
        self.previous_newline_index = -1
        self.remove_last_newline = False
        self.max_lines = max_lines
        self.newline_count = 0

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
            if new_token.count("\n"):
                self.newline_count += 1
            if self.previous_newline_index != -1:
                current_newline_index = len(self.generated_text) - 1
                substring = self.generated_text[
                    self.previous_newline_index + 1 : current_newline_index
                ]

                if substring.startswith(" ") or substring.startswith("\t"):
                    self.previous_newline_index = current_newline_index
                    if (
                        self.max_lines is not None
                        and self.newline_count >= self.max_lines
                    ):
                        return True
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
                if self.max_lines is not None and self.newline_count >= self.max_lines:
                    return True

        return False


MAX_NEW_TOKENS = 512


def generate_code_solutions(
    model,
    tokenizer,
    device,
    prompt,
    dsgi_manager,
    max_new_tokens=MAX_NEW_TOKENS,
    num_return_sequences=1,
    num_beams=None,
    inputs=None,
    nearest_future_lines=None,
    do_sample=False,
    return_full=False,
):
    if inputs is None:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Use beam search if asking for multiple completions
    if num_return_sequences > 1:
        num_beams = num_beams or num_return_sequences
        stop_criteria_list = [
            CodeGenStopCriteria(tokenizer, max_lines=nearest_future_lines)
            for _ in range(num_return_sequences)
        ]
    else:
        num_beams = 1
        stop_criteria_list = [
            CodeGenStopCriteria(tokenizer, max_lines=nearest_future_lines)
        ]

    with torch.no_grad():
        # /a/home/cc/students/cs/boazlavon/miniconda3/envs/trepan-xpy-env/lib/python3.9/site-packages/transformers/generation/utils.py
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            stopping_criteria=stop_criteria_list,
            dsgi_manager=dsgi_manager,
            do_sample=do_sample,  # Deterministic
        )

    # Collect generated texts
    generated = []
    generated_full = []
    for i, output in enumerate(outputs):
        generated_full.append(output)
        generated.append(stop_criteria_list[i].generated_text)
        # generated_full.append(tokenizer.decode(output, skip_special_tokens=True))

    # Return a list always, or a single string if only one result
    if return_full:
        return generated_full if num_return_sequences > 1 else generated_full[0]
    else:
        return generated if num_return_sequences > 1 else generated[0]
