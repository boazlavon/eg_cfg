import random
import pprint
import re
import time
import os
import json
import traceback
import random
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
from args_utils import get_dynamic_signals_str
from consts import *


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


class DsgiSessionManager:
    def __init__(self, dsgi_session_manager_args):
        self.dsgi_session_manager_args = dsgi_session_manager_args
        self.session_args = dsgi_session_manager_args.session_args
        self.guidance_args = dsgi_session_manager_args.guidance_args
        self.dynamic_signals_args = dsgi_session_manager_args.dynamic_signals_args
        self.validate_args()

    def validate_args(self):
        assert self.session_args.prompt_type in (
            PROMPT_TYPE__DEEPSEEK_BASE,
            PROMPT_TYPE__DEEPSEEK_INSTRUCT,
            PROMPT_TYPE__INSTRUCT_LONG_CODE_PROMPT,
        )

    def setup(self):
        self.solutions = {}
        self.device = setup_device()
        self.model, self.tokenizer = load_model(
            self.session_args.model_name, self.device
        )
        self.execution_manager = ExecutionManager(
            self.tokenizer, function_signature=None
        )
        self.problems = load_mbpp_problems()
        self.problems = list(self.problems.items())
        if self.session_args.is_prod:
            random.shuffle(self.problems)

    def resolve_official_evaluation_solved_entries(self):
        results_dir = self.create_results_dir(self.self.dsgi_session_manager_args)
        os.makedirs(results_dir, exist_ok=True)
        gamma = 0
        official_passed_task_ids, official_results = load_official_results(
            self.session_args.model_name
        )
        if not official_passed_task_ids:
            return
        if self.session_args.is_prod:
            random.shuffle(official_passed_task_ids)
        for task_id in official_passed_task_ids:
            for _, problem in self.problems:
                if problem["task_id"] != task_id:
                    continue
                solution_entry_path = get_solution_filepath(results_dir, task_id, gamma)
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

                if self.session_args.model_name in (
                    DEEPSEEK_13B_INSTRUCT_MODEL_NAME,
                    DEEPSEEK_CODER_V2_LITE_INSTRUCT_MODEL_NAME,
                ):
                    solution = official_results[task_id]["generation"]
                    solution_results = run_tests(solution, test_cases)
                    solution_entry = format_results(
                        solution, solution_results, general_error, tb
                    )
                    self.solutions[(task_id, gamma)] = solution_entry
                    print(solution_entry_path)
                    with open(solution_entry_path, "w") as f:
                        json.dump(solution_entry, f, indent=2)

    def resolve_cache_entries(self):
        results_dir = self.create_results_dir(self.self.dsgi_session_manager_args)
        if not (
            self.session_args.use_cache
            and self.session_args.model_name == DEEPSEEK_13B_INSTRUCT_MODEL_NAME
        ):
            return

        solved_list = DEEPSEEK_13_SOLVED_TASK_IDS
        random.shuffle(solved_list)
        time.sleep(random.randint(1, 10))
        for task_id in solved_list:
            for _, problem in self.problems:
                if problem["task_id"] != task_id:
                    continue
                print(f"Task {task_id}: Continue, Already Solved")
                for gamma in GAMMAS:
                    solution_entry_path = get_solution_filepath(
                        results_dir,
                        task_id,
                        gamma,
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
                    self.solutions[(task_id, gamma)] = solution_entry
                    with open(solution_entry_path, "w") as f:
                        json.dump(solution_entry, f, indent=2)

    def build_dsgi_injection_manager(self, problem, gamma, function_signature=None):
        test_cases = problem["test_list"]
        use_dsgi = True
        use_detector = True
        if self.session_args.prompt_type == PROMPT_TYPE__INSTRUCT_LONG_CODE_PROMPT:
            prompt, _ = format_mbpp_prompt(problem, False)
            end_string = "```"
        elif self.session_args.prompt_type in (
            PROMPT_TYPE__DEEPSEEK_BASE,
            PROMPT_TYPE__DEEPSEEK_INSTRUCT,
        ):
            if self.session_args.prompt_type == PROMPT_TYPE__DEEPSEEK_BASE:
                prompts_path = os.path.join(
                    "deepseek_mbpp_prompts", "mbpp_base_prompts.json"
                )
                end_string = "[DONE]"
            if self.session_args.prompt_type == PROMPT_TYPE__DEEPSEEK_INSTRUCT:
                prompts_path = os.path.join(
                    "deepseek_mbpp_prompts", "mbpp_instruct_prompts.json"
                )
                end_string = "```"
            with open(prompts_path, "r") as f:
                prompts = json.load(f)
                prompt = prompts[str(problem["task_id"])]

        if use_dsgi:
            dsgi_injection_manager = None
            task_kwargs = {
                "model": self.model,
                "tokenizer": self.tokenizer,
                "device": self.device,
                "function_signature": function_signature,
                "test_cases": test_cases,
                "initial_prompt": prompt,
                "dynamic_signals_types": self.guidance_args.dynamic_signals_types,
                "nf_samples_count": self.dynamic_signals_args[
                    DYNAMIC_SIGNAL__NEAREST_FUTURE_EXECUTION
                ].nf_samples_count,
                "temperature": self.dynamic_signals_args[
                    DYNAMIC_SIGNAL__NEAREST_FUTURE_EXECUTION
                ].temperature,
                "nf_samples_depth": self.dynamic_signals_args[
                    DYNAMIC_SIGNAL__NEAREST_FUTURE_EXECUTION
                ].nf_samples_depth,
                "prompt_type": self.session_args.prompt_type,
                "guidance_strategy": self.guidance_args.guidance_strategy,
                "execution_manager": self.execution_manager,
            }

            detector_kwargs = {}
            if use_detector:
                initial_prompt_input_ids_len = calculate_tokens_length(
                    self.tokenizer, prompt
                )
                function_name, _, _ = parse_mbpp_assert_statement(test_cases[0])
                detector_kwargs = {
                    "tokenizer": self.tokenizer,
                    "initial_prompt_input_ids_len": initial_prompt_input_ids_len,
                    "function_name": function_name,
                    "end_string": end_string,
                }
        task = TASK__CODE_GENERATION
        dsgi_injection_manager = DsgiInjectionManager(
            self.tokenizer,
            task,
            task_kwargs,
            gamma,
            detector_kwargs,
            use_detector=use_detector,
        )
        return dsgi_injection_manager, prompt

    def solve_problem_with_dsgi(
        self,
        problem,
        gamma,
    ):
        function_signature = None
        dsgi_injection_manager, prompt = self.build_dsgi_injection_manager(
            problem, gamma, function_signature
        )
        outputs = generate_code_solutions(
            self.model,
            self.tokenizer,
            self.device,
            prompt,
            dsgi_injection_manager,
            num_return_sequences=1,
            prompt_type=self.session_args.prompt_type,
        )
        initial_prompt_input_ids_len = calculate_tokens_length(self.tokenizer, prompt)
        new_codes = raw_outputs_to_new_code(
            outputs,
            self.tokenizer,
            initial_prompt_input_ids_len,
            self.session_args.prompt_type,
        )
        solution = new_codes[0]
        if function_signature:
            solution = f"{function_signature}\n{solution}"
            assert is_valid_python(solution)
        return solution

    def solve_problem_with_dsgi_wrapper(self, problem, gamma):
        task_id = problem["task_id"]
        test_cases = problem["test_list"]
        for retry_idx in range(self.guidance_args.retries_count):
            general_error = None
            tb = None
            # no need to solve gamma = 0 multiple times
            if gamma == 0 and retry_idx > 0:
                break
            print(f"Retry #{retry_idx + 1}")
            if self.dynamic_signals_args[
                DYNAMIC_SIGNAL__NEAREST_FUTURE_EXECUTION
            ].is_enabled:
                random.seed(40 + retry_idx)

            try:
                solution = self.solve_problem_with_dsgi(problem, gamma)
                print(solution)
            except KeyboardInterrupt:
                exit(1)
            except AssertionError as e:
                solution = None
                general_error = str(type(e))
                tb = traceback.format_exc()
                print(tb)
                if not self.session_args.is_prod:
                    raise e
            except Exception as e:
                solution = None
                general_error = str(type(e))
                tb = traceback.format_exc()
                print(tb)
                if not self.session_args.is_prod:
                    raise e

            solution_results = run_tests(solution, test_cases)
            solution_entry = format_results(
                solution, solution_results, general_error, tb
            )
            if solution_entry["passed"]:
                print(f"Problem task_id={task_id} is solved")
                break
        return solution_entry

    def create_results_dir(self, dsgi_session_manager_args):
        session_args, guidance_args, dynamic_signals_args = (
            dsgi_session_manager_args.session_args,
            dsgi_session_manager_args.guidance_args,
            dsgi_session_manager_args.dynamic_signals_args,
        )
        dynamic_signals_str = get_dynamic_signals_str(dsgi_session_manager_args)
        results_dir = os.path.join(
            "results",
            "mbpp",
            session_args.model_name.replace("/", "_"),
            dynamic_signals_str,
        )
        return results_dir

    def solve_single_problem(self, problem):
        results_dir = self.create_results_dir(self.dsgi_session_manager_args)
        task_id = problem["task_id"]
        print(f"task_id: {task_id}")
        pprint.pprint(problem)
        print()

        for gamma in self.guidance_args.gammas:
            print(f"task_id={task_id}, gamma={gamma}")
            solution_entry_path = get_solution_filepath(results_dir, task_id, gamma)
            if os.path.exists(solution_entry_path):
                print(
                    f"Solution Exists for task_id={task_id}, gamma={gamma} - continue"
                )
                continue

            if (task_id, gamma) in self.solutions:
                solution_entry = self.solutions[(task_id, gamma)]
                if "cached" in solution_entry and solution_entry["cached"]:
                    print("Problem task_id={task_id} is solved, cached")
                    break
                print(
                    f"Solution Exists for task_id={task_id}, gamma={gamma} - continue"
                )
                continue

            # touch the file to reserve it
            with open(solution_entry_path, "a"):
                pass

            solution_entry = self.solve_problem_with_dsgi_wrapper(problem, gamma)
            self.solutions[(task_id, gamma)] = solution_entry
            with open(solution_entry_path, "w") as f:
                json.dump(solution_entry, f, indent=2)
            if solution_entry["passed"]:
                print(f"Problem task_id={task_id} is solved (gamma={gamma})")
                break

    def solve(self):
        self.resolve_cache_entries()
        self.resolve_official_evaluation_solved_entries()
        for _, problem in self.problems:
            # choose configuration
            self.solve_single_problem(problem)
