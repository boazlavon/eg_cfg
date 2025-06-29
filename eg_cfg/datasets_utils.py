import re
import json
from collections import OrderedDict
import json
from datasets import load_dataset
from consts import *


def extract_function_signature(code):
    # match = re.search(r"def\s+\w+\s*\(.*\):", code)
    match = re.search(r"def\s+\w+\s*\(.*\)\s*:", code)
    return match.group(0) if match else None


def parse_assert_statement(assert_statement):
    ASSERT_PATTERN = r"^assert\s*\(*\s*(\w+)\s*\((.*)\)\s*\)*\s*==\s*(.+)$"
    match = re.match(ASSERT_PATTERN, assert_statement.strip())
    if not match:
        print(assert_statement)
        raise ValueError(f"Invalid assert statement format. {assert_statement}")

    function_name = match.group(1)
    args_str = match.group(2)
    args_str = f"({args_str})"
    expected_result_str = match.group(3)
    return (function_name, args_str, expected_result_str)


def format_simple_mbpp_prompt(problem, function_signature):
    task_description = problem["text"].strip()
    test_list = problem["test_list"]

    examples = []
    function_name = None

    for test in test_list:
        try:
            fn_name, args_str, expected = parse_assert_statement(test)
            function_name = function_name or fn_name
            examples.append(
                f"# Example {len(examples)+1}:\n# Input: {function_name}{args_str}\n# Output: {expected}"
            )
        except ValueError:
            continue

    examples_body = "\n".join(examples) if examples else ""
    examples_block = f"\n{EXAMPLES_HEADER}\n{examples_body}" if examples_body else ""

    return PROMPT_TEMPLATE.format(
        task_header=TASK_HEADER,
        text=task_description,
        goal_instruction=GOAL_INSTRUCTION,
        examples_block=examples_block,
        function_signature=function_signature,
    ).strip()


def format_test_cases_io(test_cases):
    formatted = [
        "Here are example test cases using standard input and standard output:"
    ]
    for i, (inp, out) in enumerate(test_cases, 1):
        formatted.append(f"\nTest Case {i}:")
        formatted.append(f"Input (stdin): {inp!r}")
        formatted.append(f"Expected Output (stdout): {out!r}")
    return "\n".join(formatted)


def format_long_code_prompt(problem):
    if type(problem["test_list"][0]) == str:
        test_cases = "\n".join(problem["test_list"])
    if type(problem["test_list"][0]) == tuple:
        test_cases = format_test_cases_io(problem["test_list"])
    prompt_template = LONG_CODE_INSTRUCTION_TEXT
    if test_cases:
        prompt = prompt_template.format(
            problem_text=problem["text"],
            test_cases=test_cases,
        )
    else:
        prompt_template = prompt_template.replace(TEST_CASES_INSTRUCTION, "")
        prompt = prompt_template.format(
            problem_text=problem["text"],
        )
    return prompt


def format_deepseek_instruct_mbpp_prompt(problem):
    if type(problem["test_list"][0]) == str:
        test_cases = "\n".join(problem["test_list"])
    if type(problem["test_list"][0]) == tuple:
        test_cases = format_test_cases_io(problem["test_list"])
    prompt_template = DEEPSEEK_INSTRUCT_TEMPLATE
    if test_cases:
        prompt = prompt_template.format(
            problem_text=problem["text"],
            test_cases=test_cases,
        )
        prompt = prompt.replace(">>>>", ">>>")
    else:
        prompt_template = prompt_template.replace(
            DEEPSEEK_INSTRUCT_TESTCASES_INSTRUCTION_TMP, ""
        )
        prompt = prompt_template.format(
            problem_text=problem["text"],
        )
    return prompt


def format_task_prompt(problem, deepseek_instruct=False):
    function_signature = None
    if deepseek_instruct:  # deepseek instruct prompt
        prompt = format_deepseek_instruct_mbpp_prompt(problem)
    else:  # long-code prompt
        prompt = format_long_code_prompt(problem)
    return (prompt, function_signature)


def load_mbpp_et_problems():
    test_ds = load_dataset(DATASET__MBPP_ET_HF_PATH, split="train")
    problems = OrderedDict((example["task_id"], example) for example in test_ds)
    return problems


def load_humaneval_et_problems():
    test_ds = load_dataset(DATASET__HUMANEVAL_ET_HF_PATH, split="train")
    problems = OrderedDict()
    for example in test_ds:
        task_id = example["task_id"]
        problems[task_id] = {
            "test_list": example["test_case_list"],  # list of assert statements
            "entry_point": example["entry_point"],  # function name to check
        }
    return problems


def load_mbpp_problems():
    test_ds = load_dataset(DATASET__MBPP__HF_PATH, "full", split="test")
    problems = OrderedDict((example["task_id"], example) for example in test_ds)
    return problems


def extract_asserts_for_candidate_function(test_string: str) -> list[str]:
    assert_statements = []

    # First, locate the 'def check(candidate):' block
    # This regex captures everything inside the check function, up to the next top-level statement or end of string.
    check_function_pattern = re.compile(
        r"def check\(candidate\):\s*\n(.*?)(?=\n[A-Za-z_]|\Z)", re.DOTALL
    )
    match = check_function_pattern.search(test_string)

    if match:
        check_body = match.group(1)
        # Now, find all assert statements within that body that explicitly call 'candidate('
        # ^\s*assert\s+          -> Start of line, optional leading whitespace, 'assert', one or more spaces
        # candidate\s*\(         -> The literal 'candidate' followed by optional whitespace and an opening parenthesis
        # .*$                    -> Match any characters until the end of the line
        assert_pattern = re.compile(r"^\s*assert\s+candidate\s*\(.*$", re.MULTILINE)
        found_asserts = assert_pattern.findall(check_body)

        # Strip trailing whitespace from each found assert statement
        assert_statements = [stmt.strip() for stmt in found_asserts]

    return assert_statements


def test_case_to_assert(invocation: str, expected: str) -> str:
    """Convert a (input, expected_output) pair to a Python assert statement."""
    return f"assert {invocation} == {expected}"


def load_humaneval_problems():
    with open("data/humaneval/humaneval.json", "r") as f:
        test_ds = json.load(f)
    problems = OrderedDict()
    for task_id, example in test_ds.items():
        new_example = dict(example)
        eval_test_list = extract_asserts_for_candidate_function(example["test"])
        function_signature = example["function_signature"][4:].strip()
        instruction_text = HUMANEVAL_INSTRUCTION_TEMPLATE.format(
            instruction=example["text"], function_signature=function_signature
        ).strip()
        test_cases = [
            test_case_to_assert(invocation, expected)
            for (invocation, expected) in example["test_list"]
        ]
        function_name = example["entry_point"]
        eval_tests = []
        for eval_test in eval_test_list:
            eval_test = eval_test.replace("candidate", function_name)
            eval_tests.append(eval_test)

        raw_test = example["test"]
        raw_test = f"{raw_test}\ncheck({function_name})"
        raw_test = [raw_test]

        task_id = task_id.replace("/", "_")
        new_example = {
            "task_id": task_id,
            "text": instruction_text,
            "code": example["code"],
            "test_list": test_cases,
            "eval_test_list": raw_test,
            "entry_point": function_name,
            "prompt": example["prompt"],
        }
        problems[task_id] = new_example

    return problems


def load_codecontests_problems():
    test_ds = load_dataset(DATASET__CODECONTESTS__HF_PATH, split="test")
    dataset = OrderedDict()
    for example in test_ds:
        task_id = example["name"]
        task_id = task_id.split(" ")[0].rstrip(".")
        print(task_id)
        assert len(example["public_tests"]["input"]) == len(
            example["public_tests"]["output"]
        )

        eval_test_cases = []
        test_cases = []

        for in_sample, out_sample in zip(
            example["public_tests"]["input"], example["public_tests"]["output"]
        ):
            test_cases.append((in_sample, out_sample))
        test_cases = test_cases[:TESTS_COUNT_THRESHOLD]
        for in_sample, out_sample in zip(
            example["private_tests"]["input"], example["private_tests"]["output"]
        ):
            eval_test_cases.append((in_sample, out_sample))

        for in_sample, out_sample in zip(
            example["generated_tests"]["input"],
            example["generated_tests"]["output"],
        ):
            eval_test_cases.append((in_sample, out_sample))

        # # In case there are no private tests, use public tests as eval tests
        if not eval_test_cases:
            eval_test_cases = test_cases

        function_structure_comment = SOLUTION_FUNCTION_STRUCTURE_INSTRUCTION
        function_name = "solve"
        instruction = example["description"]
        instruction = f"{instruction}\n{function_structure_comment}"
        code = EXAMPLE_CODE

        new_example = {
            "task_id": task_id,
            "text": instruction,
            "code": code,
            "test_list": test_cases,
            "eval_test_list": eval_test_cases,
            "entry_point": function_name,
            "difficulty": example["difficulty"],
        }
        dataset[task_id] = new_example
    return dataset


def load_jsonl(file_path):
    """Loads a JSON Lines file into a list of dictionaries."""
    data = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                # Remove leading/trailing whitespace and parse
                stripped_line = line.strip()
                if stripped_line:  # Ensure line is not empty
                    try:
                        data.append(json.loads(stripped_line))
                    except json.JSONDecodeError as e:
                        print(
                            f"Skipping line due to JSON decode error: {e} - Line: '{stripped_line}'"
                        )
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    return data


def load_official_results(model_name):
    official_passed_task_ids = set([])
    official_results = {}
    try:
        official_results_path = OFFICIAL_RESULT_PATH[model_name]
        official_passed_task_ids_path = OFFICIAL_PASSED_TASK_IDS_PATH[model_name]
        with open(official_passed_task_ids_path, "r") as f:
            official_passed_task_ids = set(json.load(f))
        official_results_data = load_jsonl(official_results_path)
        official_results = {}
        for entry in official_results_data:
            task_id = entry["task_id"]
            official_results[task_id] = entry
    except Exception as e:
        print(f"Error loading baseline passed IDs: {e}")
        official_passed_task_ids = set([])
        official_results = {}
    official_passed_task_ids = list(official_passed_task_ids)
    return official_passed_task_ids, official_results


LOAD_DATASET_HANDLER = {
    DATASET__MBPP: load_mbpp_problems,
    DATASET__HUMANEVAL: load_humaneval_problems,
    DATASET__CODECONTESTS: load_codecontests_problems,
    DATASET__MBPP_ET: load_mbpp_et_problems,
    DATASET__HUMANEVAL_ET: load_humaneval_et_problems,
}
