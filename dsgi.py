import random
import pprint
import black
import argparse
import os
import json
import traceback

from mbpp_utils import (
    format_mbpp_prompt,
    load_mbpp_problems,
    run_tests,
)
from model_utils import calculate_tokens_length
from mbpp_utils import parse_mbpp_assert_statement
from code_generation_utils import (
    generate_code_solutions,
    is_valid_python,
    raw_outputs_to_new_code,
)
from dsgi_manager import DsgiManager, TASK__CODE_GENERATION
from model_utils import setup_device, load_model
from code_generation_adapter import (
    DYNAMIC_SIGNAL__PARTIAL_EXECUTION,
    DYNAMIC_SIGNAL__NEAREST_FUTURE_EXECUTION,
    BACKWARD_DYNAMIC_SIGNAL_PATTERN,
    DYNAMIC_SIGNAL__BACKWARD,
    VALID_PROMPT_TYPES,
    PROMPT_TYPE__DEEPSEEK_BASE,
    PROMPT_TYPE__DEEPSEEK_INSTRUCT,
    PROMPT_TYPE__CUSTOM_PROMPT_COMPLEX,
    PROMPT_TYPE__CUSTOM_PROMPT_SIMPLE,
)
from execution_manager import ExecutionManager

MODEL_NAME = "deepseek-ai/deepseek-coder-1.3b-base"
# MODEL_NAME = "meta-llama/Llama-3.2-1B"
# MODEL_NAME = "google/gemma-3-1b-it"
# MODEL_NAME = "meta-llama/Llama-3.1-8B"
GAMMAS = (0.0, 0.1, 0.3, 0.5, 0.7, 0.8, 0.9)
FILENAME_TEMPLATE = "task_id={task_id}_gamma={gamma}.json"
FILENAME_TEMPLATE_BACKWARD_SIGNAL = (
    "task_id={task_id}_gamma={gamma}_b={backward_signals_iteration}.json"
)


def should_skip(results_dir, task_id, gamma, backward_signals_iteration):
    filepath = get_solution_filepath(
        results_dir, task_id, gamma, backward_signals_iteration
    )
    if os.path.exists(filepath):
        if backward_signals_iteration:
            print(
                f"🟡 Solution exists: task_id={task_id}, gamma={gamma} backward_iterations={backward_signals_iteration}"
            )
        else:
            print(f"🟡 Solution exists: task_id={task_id}, gamma={gamma}")
        return True

    # touch the file to reserve it
    with open(filepath, "a"):
        pass
    return False


def try_generate_code_solution(
    model,
    tokenizer,
    device,
    problem,
    gamma,
    dynamic_signals,
    backward_signals,
    prompt_type,
):
    test_cases = problem["test_list"]
    if prompt_type in (
        PROMPT_TYPE__CUSTOM_PROMPT_SIMPLE,
        PROMPT_TYPE__CUSTOM_PROMPT_COMPLEX,
    ):
        simple_prompt = PROMPT_TYPE__CUSTOM_PROMPT_SIMPLE == prompt_type
        prompt, function_signature = format_mbpp_prompt(problem, simple_prompt)
        use_dsgi = not simple_prompt
        use_detector = False

    if prompt_type in (PROMPT_TYPE__DEEPSEEK_BASE, PROMPT_TYPE__DEEPSEEK_INSTRUCT):
        if prompt_type == PROMPT_TYPE__DEEPSEEK_BASE:
            prompts_path = os.path.join(
                "deepseek_mbpp_prompts", "mbpp_base_prompts.json"
            )
            end_string = "[DONE]"
        if prompt_type == PROMPT_TYPE__DEEPSEEK_INSTRUCT:
            prompts_path = os.path.join(
                "deepseek_mbpp_prompts", "mbpp_instruct_prompts.json"
            )
            end_string = "```"
        with open(prompts_path, "r") as f:
            prompts = json.load(f)
            prompt = prompts[str(problem["task_id"])]
        use_dsgi = True
        use_detector = True
        function_signature = None

    if use_dsgi:
        dsgi_manager = None
        task_kwargs = {
            "model": model,
            "tokenizer": tokenizer,
            "device": device,
            "function_signature": function_signature,
            "test_cases": test_cases,
            "initial_prompt": prompt,
            "dynamic_signals": dynamic_signals,
            "nearest_future_samples": 2,
            "max_function_body_lines": 3,
            "backward_signals": backward_signals,
            "prompt_type": prompt_type,
        }

        detector_kwargs = {}
        if use_detector:
            initial_prompt_input_ids_len = calculate_tokens_length(tokenizer, prompt)
            function_name, _, _ = parse_mbpp_assert_statement(test_cases[0])
            detector_kwargs = {
                "tokenizer": tokenizer,
                "initial_prompt_input_ids_len": initial_prompt_input_ids_len,
                "function_name": function_name,
                "end_string": end_string,
            }
    task = TASK__CODE_GENERATION
    dsgi_manager = DsgiManager(
        tokenizer, task, task_kwargs, gamma, detector_kwargs, use_detector=use_detector
    )

    outputs = generate_code_solutions(
        model,
        tokenizer,
        device,
        prompt,
        dsgi_manager,
        num_return_sequences=1,
        prompt_type=prompt_type,
    )
    new_codes = raw_outputs_to_new_code(
        outputs, tokenizer, initial_prompt_input_ids_len, prompt_type
    )
    solution = new_codes[0]
    if function_signature:
        solution = f"{function_signature}\n{solution}"
        assert is_valid_python(solution)
    # assert dsgi_manager.detector.start_detected == 1, f"Assertion for {problem['task_id']}"
    return solution


def format_results(solution, results, general_error, tb=None):
    passed = all(r["result"] for r in results.values())
    correct = sum(int(r["result"]) for r in results.values())
    total = len(results)
    accuracy = correct / total if total else 0.0
    has_testcase_error = all([bool(result["error"]) for result in results.values()])
    entry = {
        "code": solution,
        "results": results,
        "passed": passed,
        "accuracy": accuracy,
        "general_error": general_error,
        "has_testcase_error": has_testcase_error,
    }
    if tb is not None:
        entry["tb"] = tb
    return entry


def get_solution_filepath(results_dir, task_id, gamma, backward_signals_iteration=0):
    filename = FILENAME_TEMPLATE.format(task_id=task_id, gamma=gamma)
    if backward_signals_iteration > 0:
        filename = FILENAME_TEMPLATE_BACKWARD_SIGNAL.format(
            task_id=task_id,
            gamma=gamma,
            backward_signals_iteration=backward_signals_iteration,
        )
    return os.path.join(results_dir, filename)


def extract_backward_signals(
    task_id, solutions, test_cases, tokenizer, backward_signal_iteration
):
    invalid_solutions = []
    for gamma in GAMMAS:
        solution_key = (task_id, gamma, backward_signal_iteration)
        if not solution_key in solutions:
            continue
        solution_entry = solutions[solution_key]
        if solution_entry["general_error"]:
            continue
        if solution_entry["passed"]:
            continue
        solution_code = solution_entry["code"]
        invalid_solutions.append(solution_code)
    invalid_solutions = set(invalid_solutions)

    execution_manager = ExecutionManager(tokenizer)
    # caching
    program_executions = {}
    backward_signals = []
    for code in invalid_solutions:
        program_executions[code] = execution_manager.execute_test_cases(
            code, test_cases, use_assert=True
        )

    for code in invalid_solutions:
        for test_case, program_execution in program_executions[code].items():
            trace = program_execution.to_compact_json()
            dynamic_signal = BACKWARD_DYNAMIC_SIGNAL_PATTERN.format(
                function_code=code,
                test_case=test_case,
                trace=trace,
            )
            backward_signals.append(dynamic_signal)
    return backward_signals


def generate_mbpp_solutions(
    model_name,
    results_dir,
    dynamic_signals,
    prompt_type,
    start=0,
    end=None,
    gammas=GAMMAS,
    backward_signals_iterations=0,
):
    device = setup_device()
    model, tokenizer = load_model(model_name, device)
    solutions = {}

    problems = load_mbpp_problems()
    problems = list(problems.items())

    if end is None:
        end = len(problems)
    problems = problems[start:end]
    random.shuffle(problems)

    for _, problem in problems:
        problem_solved = False
        task_id = problem["task_id"]
        test_cases = problem["test_list"]
        backward_signals = []

        print(f"task_id: {task_id}")
        pprint.pprint(problem)
        print()

        # The first iteration is always happening without a backward signal
        # so we count from the second loop of all gammas
        if DYNAMIC_SIGNAL__BACKWARD not in dynamic_signals:
            backward_signals_iterations = 0
        backward_signals_iteration = 0
        while backward_signals_iteration <= backward_signals_iterations and (
            not problem_solved
        ):
            for gamma in gammas:
                if gamma > 0 and problem_solved:
                    print(f"Skip gamma={gamma} problem is solved")
                    continue

                general_error = None
                tb = None
                solution_entry_path = get_solution_filepath(
                    results_dir, task_id, gamma, backward_signals_iteration
                )
                if os.path.exists(solution_entry_path):
                    if backward_signals_iteration:
                        print(
                            f"🟡 Solution exists: task_id={task_id}, gamma={gamma} backward_iterations={backward_signals_iteration}"
                        )
                    else:
                        print(f"🟡 Solution exists: task_id={task_id}, gamma={gamma}")
                    # load solution if invalid
                    with open(solution_entry_path, "r") as f:
                        try:
                            solution_entry = json.load(f)
                        except:
                            pass
                        else:
                            solutions[(task_id, gamma, backward_signals_iteration)] = (
                                solution_entry
                            )
                            problem_solved = solution_entry["passed"]
                    continue

                # touch the file to reserve it
                with open(solution_entry_path, "a"):
                    pass

                try:
                    solution = try_generate_code_solution(
                        model,
                        tokenizer,
                        device,
                        problem,
                        gamma,
                        dynamic_signals,
                        backward_signals,
                        prompt_type,
                    )
                    print(solution)
                except KeyboardInterrupt:
                    exit(1)
                except AssertionError as e:
                    solution = None
                    general_error = str(type(e))
                    tb = traceback.format_exc()
                    raise e
                except Exception as e:
                    solution = None
                    general_error = str(type(e))
                    tb = traceback.format_exc()
                    raise e

                solution_results = run_tests(solution, test_cases)
                solution_entry = format_results(
                    solution, solution_results, general_error, tb
                )
                with open(solution_entry_path, "w") as f:
                    json.dump(solution_entry, f, indent=2)
                solutions[(task_id, gamma, backward_signals_iteration)] = solution_entry
                if solution_entry["passed"]:
                    problem_solved = True

            if not problem_solved and DYNAMIC_SIGNAL__BACKWARD in dynamic_signals:
                # should have some counter on how many backward iterations we want. like 3 is ok
                # "general_error": general_error, "has_testcase_error": has_testcase_error,
                # For the backward pass I have a threshold of 2-3 tries.
                # I dont inject failure with errors (only failures without errors).
                # We can inject all the failure programs & their traces.
                current_backward_signals = extract_backward_signals(
                    task_id,
                    solutions,
                    test_cases,
                    tokenizer,
                    backward_signals_iteration,
                )
                backward_signals = current_backward_signals
                # backward_signals.extend(current_backward_signals)
            backward_signals_iteration += 1

    return solutions


def get_dynamic_signals(args):
    dynamic_signals_str = []
    dynamic_signals = []
    if args.p:
        dynamic_signals_str.append("p")
        dynamic_signals.append(DYNAMIC_SIGNAL__PARTIAL_EXECUTION)
    if args.n:
        dynamic_signals_str.append("n")
        dynamic_signals.append(DYNAMIC_SIGNAL__NEAREST_FUTURE_EXECUTION)
    if args.b:
        dynamic_signals_str.append("b")
        dynamic_signals.append(DYNAMIC_SIGNAL__BACKWARD)
    dynamic_signals_str = "".join(dynamic_signals_str)
    dynamic_signals = tuple(dynamic_signals)
    assert dynamic_signals
    return dynamic_signals, dynamic_signals_str


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name", type=str, default=MODEL_NAME, help="Name of the model used"
    )
    parser.add_argument("--b", action="store_true", help="Enable backward signal")
    parser.add_argument("--n", action="store_true", help="Enable nearest future signal")
    parser.add_argument(
        "--p",
        action="store_true",
        help="Enable partial execution signal (default: enabled)",
    )
    parser.add_argument(
        "--prompt-type",
        type=str,
        required=True,
        choices=VALID_PROMPT_TYPES,
        help="Type of prompt to use. Must be one of: " + ", ".join(VALID_PROMPT_TYPES),
    )

    args = parser.parse_args()

    # Create results directory
    sanitized_model_name = args.model_name.replace("/", "_")
    dynamic_signals, dynamic_signals_str = get_dynamic_signals(args)
    results_dir = os.path.join(
        "results", "mbpp", sanitized_model_name, dynamic_signals_str
    )
    # results_dir = os.path.join("results", "mbpp", sanitized_model_name)
    os.makedirs(results_dir, exist_ok=True)
    backward_signals_iterations = 0
    if DYNAMIC_SIGNAL__BACKWARD in dynamic_signals:
        backward_signals_iterations = 2

    prompt_type = args.prompt_type
    model_name = args.model_name
    generate_mbpp_solutions(
        model_name,
        results_dir,
        dynamic_signals,
        prompt_type,
        backward_signals_iterations=backward_signals_iterations,
    )


if __name__ == "__main__":
    main()
