import random
import pprint
import black
import argparse
import os
import json

from mbpp_utils import (
    format_mbpp_prompt,
    load_mbpp_problems,
    run_tests,
)
from code_generation_utils import generate_code_solutions, is_valid_python
from dsgi_manager import DsgiManager, TASK__CODE_GENERATION
from model_utils import setup_device, load_model
from code_generation_adapter import (
    DYNAMIC_SIGNAL__PARTIAL_EXECUTION,
    DYNAMIC_SIGNAL__NEAREST_FUTURE_EXECUTION,
    BACKWARD_DYNAMIC_SIGNAL_PATTERN,
    DYNAMIC_SIGNAL__BACKWARD,
)
from execution_manager import ExecutionManager

MODEL_NAME = "meta-llama/Llama-3.2-1B"
# MODEL_NAME = "google/gemma-3-1b-it"
# MODEL_NAME = "meta-llama/Llama-3.1-8B"
NO_GUIDANCE_SIMPLE_PROMPT_GAMMA = -0.1
# GAMMAS = (NO_GUIDANCE_SIMPLE_PROMPT_GAMMA, 0.0, 0.1, 0.3, 0.5, 0.7, 0.8, 0.9)
GAMMAS = (
    0.0,
    0.8,
)
FILENAME_TEMPLATE = "task_id={task_id}_gamma={gamma}.json"
FILENAME_TEMPLATE_BACKWARD_SIGNAL = (
    "task_id={task_id}_gamma={gamma}_b={backward_signals_iteration}.json"
)


def should_skip(results_dir, task_id, gamma, backward_signals_iteration):
    filepath = get_solution_filepath(
        results_dir, task_id, gamma, backward_signals_iteration
    )
    if os.path.exists(filepath):
        print(f"🟡 Solution exists: task_id={task_id}, gamma={gamma}")
        return True
    # touch the file to reserve it
    with open(filepath, "a"):
        pass
    return False


def try_generate_code_solutions(
    model, tokenizer, device, problem, gamma, dynamic_signals, backward_signals
):
    simple_prompt = gamma == NO_GUIDANCE_SIMPLE_PROMPT_GAMMA
    prompt, function_signature = format_mbpp_prompt(problem, simple_prompt)
    test_cases = problem["test_list"]

    dsgi_manager = None
    if not simple_prompt:
        task_kwargs = {
            "model": model,
            "tokenizer": tokenizer,
            "device": device,
            "function_signature": function_signature,
            "test_cases": test_cases,
            "initial_prompt": prompt,
            "dynamic_signals": dynamic_signals,
            "nearest_future_samples": 5,
            "nearest_future_lines": 3,
            "backward_signals": backward_signals,
        }
        task = TASK__CODE_GENERATION
        dsgi_manager = DsgiManager(tokenizer, task, task_kwargs, gamma)

    function_code = generate_code_solutions(
        model, tokenizer, device, prompt, dsgi_manager
    )
    solution = f"{function_signature}\n{function_code}"

    assert is_valid_python(solution)
    solution = black.format_str(solution, mode=black.FileMode(line_length=1024))
    return solution


def format_results(solution, results, general_error):
    passed = all(r["result"] for r in results.values())
    correct = sum(int(r["result"]) for r in results.values())
    total = len(results)
    accuracy = correct / total if total else 0.0
    has_testcase_error = all([bool(result["error"]) for result in results.values()])
    return {
        "code": solution,
        "results": results,
        "passed": passed,
        "accuracy": accuracy,
        "general_error": general_error,
        "has_testcase_error": has_testcase_error,
    }


def get_solution_filepath(results_dir, task_id, gamma, backward_signals_iteration):
    filename = FILENAME_TEMPLATE.format(task_id=task_id, gamma=gamma)
    if backward_signals_iteration > 0:
        filename = FILENAME_TEMPLATE_BACKWARD_SIGNAL.format(
            task_id=task_id,
            gamma=gamma,
            backward_signals_iteration=backward_signals_iteration,
        )
    return os.path.join(results_dir, filename)


def extract_backward_signals(task_id, solutions, test_cases, tokenizer):
    invalid_solutions = []
    for gamma in GAMMAS:
        solution_entry = solutions[(task_id, gamma)]
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
    results_dir,
    dynamic_signals,
    start=0,
    end=None,
    gammas=GAMMAS,
    backward_signals_iterations=0,
):
    device = setup_device()
    model, tokenizer = load_model(MODEL_NAME, device)
    solutions = {}

    problems = load_mbpp_problems()
    problems = list(problems.items())

    if end is None:
        end = len(problems)
    problems = problems[start:end]
    # random.shuffle(problems)

    for _, problem in problems:
        problem_solved = False
        task_id = problem["task_id"]
        if 11 == int(task_id):
            continue
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
                if should_skip(results_dir, task_id, gamma, backward_signals_iteration):
                    continue

                try:
                    solution = try_generate_code_solutions(
                        model,
                        tokenizer,
                        device,
                        problem,
                        gamma,
                        dynamic_signals,
                        backward_signals,
                    )
                    print(solution)
                except KeyboardInterrupt:
                    exit(1)
                except AssertionError as e:
                    solution = None
                    general_error = str(type(e))
                    raise e
                except Exception as e:
                    solution = None
                    general_error = str(type(e))
                    raise e

                print()
                solution_results = run_tests(solution, test_cases)
                solution_entry = format_results(
                    solution, solution_results, general_error
                )
                filepath = get_solution_filepath(results_dir, task_id, gamma)
                with open(filepath, "w") as f:
                    json.dump(solution_entry, f, indent=2)
                solutions[(task_id, gamma)] = solution_entry
                if solution_entry["passed"]:
                    problem_solved = True

            backward_signals_iteration += 1
            if not problem_solved and DYNAMIC_SIGNAL__BACKWARD in dynamic_signals:
                # should have some counter on how many backward iterations we want. like 3 is ok
                # "general_error": general_error, "has_testcase_error": has_testcase_error,
                # For the backward pass I have a threshold of 2-3 tries.
                # I dont inject failure with errors (only failures without errors).
                # We can inject all the failure programs & their traces.
                current_backward_signals = extract_backward_signals(
                    task_id, solutions, test_cases, tokenizer
                )
                backward_signals.extend(current_backward_signals)

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
        default=True,
        help="Enable partial execution signal (default: enabled)",
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

    generate_mbpp_solutions(
        results_dir,
        dynamic_signals,
        backward_signals_iterations=backward_signals_iterations,
    )


if __name__ == "__main__":
    main()
