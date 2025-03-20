import ast
import re
import black


def make_executable(prompt: str, code: str, fallback_to_prompt: bool = True) -> str:
    lines = code.split("\n")
    fixed_code = ""

    while lines:
        fixed_code = "\n".join(lines)
        if is_valid_python(fixed_code) and fixed_code.startswith(prompt):
            break

        # Remove last line and try again
        last_line = lines.pop()
        if not lines:
            break  # Stop if there are no lines left

        fixed_code = "\n".join(lines)
        if is_valid_python(fixed_code) and fixed_code.startswith(prompt):
            break

        # If removing doesn't work, replace last line with 'pass' (preserving indentation)
        indent = re.match(r"\s*", last_line).group(0)
        lines.append(f"{indent}pass")
        fixed_code = "\n".join(lines)
        if is_valid_python(fixed_code) and fixed_code.startswith(prompt):
            break
        lines.pop()  # Remove the pass if it's still invalid

    if (
        not is_valid_python(fixed_code) or not fixed_code.startswith(prompt)
    ) and fallback_to_prompt:
        prompt_lines = prompt.split("\n")
        last_line = prompt_lines[-1]
        indent = re.match(r"\s*", last_line).group(0)
        fixed_code = f"{prompt}"
        fixed_code = black.format_str(fixed_code, mode=black.FileMode())

    return fixed_code


def make_executable_partial_code(prompt, new_tokens):
    # Randomly cut the generated code at some point
    for cut_index in range(len(new_tokens)):
        cut_code = f"{prompt}{new_tokens[:cut_index]}"
        executable_code = make_executable(prompt, cut_code)
        print(executable_code)


def is_valid_python(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False
