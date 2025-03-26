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


MAX_NEW_TOKENS = 512


def generate_code_solution(
    model, tokenizer, device, prompt, dsgi_manager, max_new_tokens=MAX_NEW_TOKENS
):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    stop_criteria = CodeGenStopCriteria(tokenizer)

    with torch.no_grad():
        # /a/home/cc/students/cs/boazlavon/miniconda3/envs/trepan-xpy-env/lib/python3.9/site-packages/transformers/generation/utils.py
        model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            stopping_criteria=[stop_criteria],
            dsgi_manager=dsgi_manager,
            do_sample=False,
        )

    # _ = tokenizer.decode(outputs[0], skip_special_tokens=True)
    new_tokens = stop_criteria.generated_text
    return new_tokens
