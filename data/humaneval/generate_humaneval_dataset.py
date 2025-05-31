import ast
import json
from typing import List, Tuple
from datasets import load_dataset
import ast
from typing import List, Tuple
from mbpp_utils import run_tests, evaluate_solution
from eg_cfg_session_manager import format_results

import ast
import re
import ast
import inspect


def extract_signature_by_line(code: str, entry_point: str) -> str:
    """
    Finds the first line in the code that starts with 'def {entry_point}' and returns it without the trailing colon.

    Args:
        code (str): Full Python source code.
        entry_point (str): Function name to search for.

    Returns:
        str: Function signature line without the colon.
    """
    for line in code.splitlines():
        line = line.strip()
        if line.startswith(f"def {entry_point}"):
            return line.rstrip(":").strip()
    raise ValueError(f"Function '{entry_point}' not found.")


def extract_instruction_and_tests_clean(
    prompt_code: str, entry_point: str
) -> Tuple[str, List[Tuple[str, str]], str]:
    """
    Extracts instruction and test cases from prompt docstring.
    Ensures:
      - Instruction contains no '>>>'
      - Test case inputs/results contain no '>>>'

    Returns:
        - instruction: str
        - test_cases: List of (invocation, result)
        - cleaned_prompt: str (prompt with instruction removed from docstring)
    """
    # Parse the function
    tree = ast.parse(prompt_code)
    func_node = next(
        (
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == entry_point
        ),
        None,
    )
    if not func_node:
        raise ValueError(f"Function '{entry_point}' not found.")

    # Extract full (raw) docstring
    docstring = ast.get_docstring(func_node, clean=False)
    if not docstring:
        return "", [], prompt_code

    # 1. Extract instruction (everything before first >>>)
    split_doc = docstring.split(">>>", maxsplit=1)
    instruction = split_doc[0].strip()
    assert ">>>" not in instruction, "Instruction must not contain '>>>'"

    remaining = split_doc[1] if len(split_doc) > 1 else ""

    # 2. Extract test cases from the remaining docstring
    test_cases = []
    for segment in remaining.split(">>>"):
        lines = segment.strip().splitlines()
        if not lines:
            continue
        invocation = lines[0].strip()
        result = lines[1].strip() if len(lines) > 1 else ""

        assert ">>>" not in invocation, "Invocation must not contain '>>>'"
        assert ">>>" not in result, "Result must not contain '>>>'"

        test_cases.append((invocation, result))

    # 3. Clean the prompt by removing the instruction from the docstring
    new_docstring = '"""\n' + "\n>>>".join([""] + remaining.split(">>>")) + '\n"""'
    updated_prompt = prompt_code.replace(docstring, new_docstring)

    return instruction, test_cases, updated_prompt


def test_case_to_assert(invocation: str, expected: str) -> str:
    """Convert a (input, expected_output) pair to a Python assert statement."""
    return f"assert {invocation} == {expected}"


def main():
    dataset = None
    try:
        with open("humaneval.json", "r") as f:
            dataset = json.load(f)
    except:
        pass

    if not dataset:
        humaneval = load_dataset("openai/openai_humaneval")
        extracted_components = {
            example["task_id"]: extract_instruction_and_tests_clean(
                example["prompt"], example["entry_point"]
            )
            for example in humaneval["test"]
            if len(
                extract_instruction_and_tests_clean(
                    example["prompt"], example["entry_point"]
                )[1]
            )
            == example["prompt"].count(">>>")
        }
        assert len(extracted_components) == len(humaneval["test"])
        dataset = {}
        for example in humaneval["test"]:
            task_id = example["task_id"]
            print(task_id)
            dataset[task_id] = example
            instruction, test_cases, _ = extracted_components[task_id]
            function_sig = extract_signature_by_line(
                example["prompt"], example["entry_point"]
            )
            solution = example["prompt"] + example["canonical_solution"]
            instruction = instruction.replace("\n", "")
            instruction = instruction.replace("    ", " ")
            instruction = instruction.replace("   ", " ")
            dataset[task_id]["text"] = instruction
            dataset[task_id]["function_signature"] = function_sig
            assert function_sig in example["prompt"]
            dataset[task_id]["test_list"] = test_cases
            dataset[task_id]["code"] = solution

        with open("humaneval.json", "w") as f:
            json.dump(dataset, f, indent=2)

    solutions = {}
    for task_id, example in dataset.items():
        task_id = example["task_id"]
        print(task_id)
        test_cases = [
            test_case_to_assert(invocation, expected)
            for (invocation, expected) in example["test_list"]
        ]
        solution = example["prompt"] + example["canonical_solution"]
        solution_results = run_tests(solution, test_cases)
        general_error = None
        tb = None
        solution_entry = format_results(solution, solution_results, general_error, tb)
        solutions[task_id] = solution_entry

    invalid_entries = {
        task_id
        for task_id, solution_entry in solutions.items()
        if not solution_entry["passed"]
    }
    print()
    print(f"Total: {len(solutions)}")
    print(f"Valid Entries: {len(solutions) - len(invalid_entries)}")

    # HumanEval116 canoncial solution is wrong, so its an invalid entry
    # HumanEval47 has a wrong test case

    # This samples are using == instead of newline so we manually parsed their test-cases but can easily done with regex
    # {'HumanEval/113', 'HumanEval/145', 'HumanEval/156', 'HumanEval/108', 'HumanEval/128', 'HumanEval/12', 'HumanEval/162'}
    print(f"Invalid Entries: {len(invalid_entries)}")
    print(invalid_entries)


main()
