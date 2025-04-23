import random
import pprint
import time
import os
import json
import traceback
import random
from args_utils import get_cmdline_args, build_dsgi_session_manager_args

from mbpp_utils import (
    format_mbpp_prompt,
    load_mbpp_problems,
    run_tests,
)
from model_utils import calculate_tokens_length
from mbpp_utils import parse_mbpp_assert_statement, load_official_results
from code_generation_utils import (
    generate_code_solutions,
    is_valid_python,
    raw_outputs_to_new_code,
)
from dsgi_injection_manager import DsgiInjectionManager
from model_utils import setup_device, load_model
from execution_manager import ExecutionManager
from dsgi_session_manager import (
    DsgiSessionManager,
    format_results,
    get_solution_filepath,
)

from consts import *


def try_generate_code_solution(
    model,
    tokenizer,
    device,
    problem,
    gamma,
    dynamic_signals,
    backward_signals,
    prompt_type,
    nf_samples_count,
    temperature,
    nf_samples_depth,
    guidance_strategy,
):
    test_cases = problem["test_list"]
    if prompt_type in (
        PROMPT_TYPE__CUSTOM_PROMPT_SIMPLE,
        PROMPT_TYPE__INSTRUCT_LONG_CODE_PROMPT,
    ):
        assert (
            prompt_type != PROMPT_TYPE__CUSTOM_PROMPT_SIMPLE
        ), "Unsupported Prompt Type"
        # prompt, function_signature = format_mbpp_prompt(problem, simple_prompt)
        prompt, _ = format_mbpp_prompt(problem, False)
        use_dsgi = True
        use_detector = True
        function_signature = None
        end_string = "```"

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
        dsgi_injection_manager = None
        task_kwargs = {
            "model": model,
            "tokenizer": tokenizer,
            "device": device,
            "function_signature": function_signature,
            "test_cases": test_cases,
            "initial_prompt": prompt,
            "dynamic_signals": dynamic_signals,
            "nf_samples_count": nf_samples_count,
            "temperature": temperature,
            "nf_samples_depth": nf_samples_depth,
            "backward_signals": backward_signals,
            "prompt_type": prompt_type,
            "guidance_strategy": guidance_strategy,
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
    dsgi_injection_manager = DsgiInjectionManager(
        tokenizer, task, task_kwargs, gamma, detector_kwargs, use_detector=use_detector
    )

    outputs = generate_code_solutions(
        model,
        tokenizer,
        device,
        prompt,
        dsgi_injection_manager,
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
    # assert dsgi_injection_manager.detector.start_detected == 1, f"Assertion for {problem['task_id']}"
    return solution


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
    dynamic_signals_types,
    prompt_type,
    start=0,
    end=None,
    gammas=GAMMAS,
    backward_signals_iterations=0,
    prod=False,
    nf_samples_count=None,
    temperature=None,
    nf_samples_depth=None,
    retries_count=1,
    guidance_strategy=GUIDANCE_STRATEGY__TOKEN_GUIDANCE,
    use_cache=False,
):
    device = setup_device()
    model, tokenizer = load_model(model_name, device)
    solutions = {}
    official_passed_task_ids, official_results = load_official_results(model_name)

    problems = load_mbpp_problems()
    problems = list(problems.items())
    if end is None:
        end = len(problems)
    problems = problems[start:end]

    if prod:
        random.shuffle(problems)
        random.shuffle(official_passed_task_ids)

    backward_signals_iteration = 0
    if use_cache and model_name == DEEPSEEK_13B_INSTRUCT_MODEL_NAME:
        solved_list = DEEPSEEK_13_SOLVED_TASK_IDS
        random.shuffle(solved_list)
        time.sleep(random.randint(1, 10))
        for task_id in solved_list:
            for _, problem in problems:
                if problem["task_id"] != task_id:
                    continue
                print(f"Task {task_id}: Continue, Already Solved")
                for gamma in GAMMAS:
                    solution_entry_path = get_solution_filepath(
                        results_dir, task_id, gamma, backward_signals_iteration
                    )
                    if os.path.exists(solution_entry_path):
                        continue
                    with open(solution_entry_path, "a"):
                        pass
                    solution_entry = {
                        "code": "",
                        "results": {},
                        "passed": gamma > 0,
                        "accuracy": float(int(gamma > 0)),
                        "general_error": None,
                        "has_testcase_error": False,
                        "cached": True,
                    }
                    solutions[(task_id, gamma, backward_signals_iteration)] = (
                        solution_entry
                    )
                    with open(solution_entry_path, "w") as f:
                        json.dump(solution_entry, f, indent=2)

    gamma = 0
    backward_signals_iteration = 0
    if official_passed_task_ids:
        for task_id in official_passed_task_ids:
            for _, problem in problems:
                if problem["task_id"] != task_id:
                    continue
                solution_entry_path = get_solution_filepath(
                    results_dir, task_id, gamma, backward_signals_iteration
                )
                if os.path.exists(solution_entry_path):
                    continue
                with open(solution_entry_path, "a"):
                    pass
                print(f"task_id: {task_id}")
                pprint.pprint(problem)
                test_cases = problem["test_list"]

                solution = None
                general_error = None
                tb = None
                solution_entry = None

                if model_name in (
                    DEEPSEEK_13B_INSTRUCT_MODEL_NAME,
                    DEEPSEEK_CODER_V2_LITE_INSTRUCT_MODEL_NAME,
                ):
                    solution = official_results[task_id]["generation"]
                    solution_results = run_tests(solution, test_cases)
                    solution_entry = format_results(
                        solution, solution_results, general_error, tb
                    )
                    solutions[(task_id, gamma, backward_signals_iteration)] = (
                        solution_entry
                    )
                    print(solution_entry_path)
                    with open(solution_entry_path, "w") as f:
                        json.dump(solution_entry, f, indent=2)

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
        if DYNAMIC_SIGNAL__BACKWARD not in dynamic_signals_types:
            backward_signals_iterations = 0
        backward_signals_iteration = 0
        while backward_signals_iteration <= backward_signals_iterations and (
            not problem_solved
        ):
            for gamma in gammas:
                print(f"gamma={gamma}")
                if gamma > 0 and problem_solved:
                    print(f"Skip gamma={gamma} problem is solved")
                    break

                if (task_id, gamma, backward_signals_iteration) in solutions:
                    solution_entry = solutions[
                        (task_id, gamma, backward_signals_iteration)
                    ]
                    if "cached" in solution_entry and solution_entry["cached"]:
                        problem_solved = solution_entry["passed"]
                        print("cached")
                        continue

                solution_entry_path = get_solution_filepath(
                    results_dir, task_id, gamma, backward_signals_iteration
                )
                if os.path.exists(solution_entry_path):
                    print("Solution Exists")
                    continue
                    # if backward_signals_iteration:
                    #     print(
                    #         f"🟡 Solution exists: task_id={task_id}, gamma={gamma} backward_iterations={backward_signals_iteration}"
                    #     )
                    # else:
                    #     print(
                    #         f"🟡 Solution exists: task_id={task_id}, gamma={gamma}"
                    #     )
                    # load solution if invalid
                    # with open(solution_entry_path, "r") as f:
                    #     try:
                    #         solution_entry = json.load(f)
                    #     except:
                    #         pass
                    #     else:
                    #         solutions[
                    #             (task_id, gamma, backward_signals_iteration)
                    #         ] = solution_entry
                    #         problem_solved = solution_entry["passed"]

                # touch the file to reserve it
                with open(solution_entry_path, "a"):
                    pass

                for attempt_idx in range(retries_count):
                    general_error = None
                    tb = None
                    if gamma == 0 and attempt_idx > 0:
                        break
                    print(f"Attempt #{attempt_idx + 1}")
                    if (
                        DYNAMIC_SIGNAL__NEAREST_FUTURE_EXECUTION
                        in dynamic_signals_types
                    ):
                        random.seed(40 + attempt_idx)

                    try:
                        solution = try_generate_code_solution(
                            model,
                            tokenizer,
                            device,
                            problem,
                            gamma,
                            dynamic_signals_types,
                            backward_signals,
                            prompt_type,
                            nf_samples_count,
                            temperature,
                            nf_samples_depth,
                            guidance_strategy,
                        )
                        print(solution)
                    except KeyboardInterrupt:
                        exit(1)
                    except AssertionError as e:
                        solution = None
                        general_error = str(type(e))
                        tb = traceback.format_exc()
                        print(tb)
                        if not prod:
                            raise e
                    except Exception as e:
                        solution = None
                        general_error = str(type(e))
                        tb = traceback.format_exc()
                        print(tb)
                        if not prod:
                            raise e
                    solution_results = run_tests(solution, test_cases)
                    solution_entry = format_results(
                        solution, solution_results, general_error, tb
                    )
                    if solution_entry["passed"]:
                        problem_solved = True
                        break
                solutions[(task_id, gamma, backward_signals_iteration)] = solution_entry
                with open(solution_entry_path, "w") as f:
                    json.dump(solution_entry, f, indent=2)

            if not problem_solved and DYNAMIC_SIGNAL__BACKWARD in dynamic_signals_types:
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


def main():
    generate_mbpp_solutions(
        args.model_name,
        results_dir,
        dynamic_signals_types,
        args.prompt_type,
        backward_signals_iterations=backward_signals_iterations,
        prod=args.prod,
        nf_samples_count=args.s,
        temperature=args.t,
        retries_count=args.r,
        nf_samples_depth=args.d,
        guidance_strategy=args.g,
        use_cache=args.cache,
    )


if __name__ == "__main__":
    main()
