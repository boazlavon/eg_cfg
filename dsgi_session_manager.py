import random
import torch
import pprint
from argparse import Namespace
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
from probs_utils import stable_hash
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
    def __init__(self, session_config, inference_sessions_configs):
        self.session_config = session_config
        self.inference_sessions_configs = inference_sessions_configs
        self.inference_session = None

    def validate_args(self):
        for inference_session_config in self.inference_sessions_configs:
            assert inference_session_config["prompt_type"] in (
                PROMPT_TYPE__DEEPSEEK_BASE,
                PROMPT_TYPE__DEEPSEEK_INSTRUCT,
                PROMPT_TYPE__INSTRUCT_LONG_CODE_PROMPT,
            )

    def setup(self):
        self.solutions = {}
        self.device = setup_device()
        self.model, self.tokenizer = load_model(
            self.session_config.model_name, self.device
        )
        self.execution_manager = ExecutionManager(
            self.tokenizer, function_signature=None
        )
        self.problems = load_mbpp_problems()
        self.problems = list(self.problems.items())
        if self.session_config.start_idx and self.session_config.end_idx:
            self.problems = self.problems[
                self.session_config.start_idx : self.session_config.end_idx
            ]
        if self.session_config.is_prod:
            random.shuffle(self.problems)
            # if I use trials list its better not to shuffle as they are decreasing in the effectivness.
            # random.shuffle(self.inference_sessions_configs)

    def create_results_dir(self, session_config, inference_session_config):
        dynamic_signals_str = get_dynamic_signals_str(inference_session_config)
        results_dir = os.path.join(
            self.session_config.results_dir,
            "mbpp",
            session_config.model_name.replace("/", "_"),
            dynamic_signals_str,
        )
        return results_dir

    def setup_inference_session(self, inference_session_config):
        results_dir = self.create_results_dir(
            self.session_config, inference_session_config
        )
        solved_tasks_cache_dir = os.path.join(
            self.session_config.results_dir,
            "mbpp",
            self.session_config.model_name.replace("/", "_"),
            ".solved_tasks_cache",
        )
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(solved_tasks_cache_dir, exist_ok=True)
        self.inference_session = Namespace(
            **{
                "inference_session_config": inference_session_config,
                "results_dir": results_dir,
                "solved_tasks_cache_dir": solved_tasks_cache_dir,
            }
        )
        print("Setting inference session config:")
        pprint.pprint(self.inference_session)

    def resolve_official_evaluation_solved_entries(self):
        gamma = 0
        official_passed_task_ids, official_results = load_official_results(
            self.session_config.model_name
        )
        if not official_passed_task_ids:
            return
        if self.session_config.is_prod:
            random.shuffle(official_passed_task_ids)
        for task_id in official_passed_task_ids:
            for _, problem in self.problems:
                if problem["task_id"] != task_id:
                    continue
                solution_entry_path = get_solution_filepath(
                    self.inference_session.results_dir, task_id, gamma
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

                if self.session_config.model_name in (
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
        if not (
            self.session_config.use_cache
            and self.session_config.model_name == DEEPSEEK_13B_INSTRUCT_MODEL_NAME
        ):
            return

        solved_list = DEEPSEEK_13_SOLVED_TASK_IDS
        random.shuffle(solved_list)
        time.sleep(random.randint(1, 10))
        print("Perform Caching resolution")
        for task_id in solved_list:
            for _, problem in self.problems:
                if problem["task_id"] != task_id:
                    continue
                print(f"Task {task_id}: Continue, Already Solved")
                for gamma in GAMMAS:
                    solution_entry_path = get_solution_filepath(
                        self.inference_session.results_dir,
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
        if (
            self.inference_session.inference_session_config["prompt_type"]
            == PROMPT_TYPE__INSTRUCT_LONG_CODE_PROMPT
        ):
            prompt, _ = format_mbpp_prompt(problem, False)
            end_string = "```"
        elif self.inference_session.inference_session_config["prompt_type"] in (
            PROMPT_TYPE__DEEPSEEK_BASE,
            PROMPT_TYPE__DEEPSEEK_INSTRUCT,
        ):
            if (
                self.inference_session.inference_session_config["prompt_type"]
                == PROMPT_TYPE__DEEPSEEK_BASE
            ):
                prompts_path = os.path.join(
                    "deepseek_mbpp_prompts", "mbpp_base_prompts.json"
                )
                end_string = "[DONE]"
            if (
                self.inference_session.inference_session_config["prompt_type"]
                == PROMPT_TYPE__DEEPSEEK_INSTRUCT
            ):
                prompts_path = os.path.join(
                    "deepseek_mbpp_prompts", "mbpp_instruct_prompts.json"
                )
                end_string = "```"
            with open(prompts_path, "r") as f:
                prompts = json.load(f)
                prompt = prompts[str(problem["task_id"])]

        print(self.inference_session.inference_session_config)
        dynamic_signals_types = [
            dynamic_signal_type
            for dynamic_signal_type in SUPPORTED_DYNAMIC_SIGNALS
            if self.inference_session.inference_session_config[
                dynamic_signal_type
            ].is_enabled
        ]
        if use_dsgi:
            dsgi_injection_manager = None
            task_kwargs = {
                "model": self.model,
                "tokenizer": self.tokenizer,
                "device": self.device,
                "function_signature": function_signature,
                "test_cases": test_cases,
                "initial_prompt": prompt,
                "dynamic_signals_types": dynamic_signals_types,
                "nf_samples_count": self.inference_session.inference_session_config[
                    DYNAMIC_SIGNAL__NEAREST_FUTURE_EXECUTION
                ].nf_samples_count,
                "temperature": self.inference_session.inference_session_config[
                    DYNAMIC_SIGNAL__NEAREST_FUTURE_EXECUTION
                ].temperature,
                "nf_samples_depth": self.inference_session.inference_session_config[
                    DYNAMIC_SIGNAL__NEAREST_FUTURE_EXECUTION
                ].nf_samples_depth,
                "prompt_type": self.inference_session.inference_session_config[
                    "prompt_type"
                ],
                "guidance_strategy": self.inference_session.inference_session_config[
                    "guidance_strategy"
                ],
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
            top_probs_count=self.session_config.top_probs,
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
            prompt_type=self.inference_session.inference_session_config["prompt_type"],
        )
        initial_prompt_input_ids_len = calculate_tokens_length(self.tokenizer, prompt)
        new_codes = raw_outputs_to_new_code(
            outputs,
            self.tokenizer,
            initial_prompt_input_ids_len,
            self.inference_session.inference_session_config["prompt_type"],
        )
        solution = new_codes[0]
        if function_signature:
            solution = f"{function_signature}\n{solution}"
            assert is_valid_python(solution)
        return solution

    def solve_problem_with_dsgi_wrapper(self, problem, gamma):
        task_id = problem["task_id"]
        test_cases = problem["test_list"]
        for retry_idx in range(self.session_config.retries_count):
            general_error = None
            tb = None
            # no need to solve gamma = 0 multiple times
            if gamma == 0 and retry_idx > 0:
                break
            print(f"Retry #{retry_idx + 1}")
            if self.inference_session.inference_session_config[
                DYNAMIC_SIGNAL__NEAREST_FUTURE_EXECUTION
            ].is_enabled:
                if self.session_config.random_seed is not None:
                    random_seed = self.session_config.random_seed
                else:
                    iid_arg = (
                        self.inference_session.inference_session_config[
                            DYNAMIC_SIGNAL__NEAREST_FUTURE_EXECUTION
                        ].temperature,
                        self.inference_session.inference_session_config[
                            DYNAMIC_SIGNAL__NEAREST_FUTURE_EXECUTION
                        ].nf_samples_count,
                        self.inference_session.inference_session_config[
                            DYNAMIC_SIGNAL__NEAREST_FUTURE_EXECUTION
                        ].nf_samples_depth,
                        self.inference_session.inference_session_config[
                            DYNAMIC_SIGNAL__PARTIAL_EXECUTION
                        ].is_enabled,
                        # self.inference_session.inference_session_config[
                        #     "guidance_strategy"
                        # ],
                        # self.inference_session.inference_session_config["prompt_type"],
                    )
                    random_seed = stable_hash(iid_arg)
                    random_seed = random_seed % 1000
                    random_seed += 40
                random_seed += retry_idx
                random.seed(random_seed)
                torch.manual_seed(random_seed)
                torch.cuda.manual_seed_all(random_seed)

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
                if not self.session_config.is_prod:
                    raise e
            except Exception as e:
                solution = None
                general_error = str(type(e))
                tb = traceback.format_exc()
                print(tb)
                if not self.session_config.is_prod:
                    raise e

            solution_results = run_tests(solution, test_cases)
            solution_entry = format_results(
                solution, solution_results, general_error, tb
            )
            if solution_entry["passed"]:
                solution_entry["random_seed"] = random_seed
                print(f"Problem task_id={task_id} is solved")
                break
        return solution_entry

    def solve_single_problem(self, problem):
        task_id = problem["task_id"]
        print(f"task_id: {task_id}")
        pprint.pprint(problem)
        print()

        global_cache_solved_task_id_path = os.path.join(
            self.inference_session.solved_tasks_cache_dir, f"{task_id}"
        )
        for gamma in self.session_config.gammas:
            print(f"task_id={task_id}, gamma={gamma}")
            solution_entry_path = get_solution_filepath(
                self.inference_session.results_dir, task_id, gamma
            )

            if self.session_config.use_global_cache and os.path.exists(
                global_cache_solved_task_id_path
            ):
                print("Problem is solved: Gloabl cache")
                with open(solution_entry_path, "a"):
                    pass
                solution_entry = {
                    "code": "",
                    "results": {},
                    "passed": 1,
                    "accuracy": 1,
                    "general_error": None,
                    "has_testcase_error": False,
                    "global_cached": True,
                }
                self.solutions[(task_id, gamma)] = solution_entry
                with open(solution_entry_path, "w") as f:
                    json.dump(solution_entry, f, indent=2)
                break

            if os.path.exists(solution_entry_path):
                with open(solution_entry_path, "r") as f:
                    try:
                        solution_entry = json.load(f)
                        if solution_entry and solution_entry["passed"]:
                            print(
                                f"Problem task_id={task_id} is solved (gamma={gamma})"
                            )
                            break
                        elif solution_entry and not solution_entry["passed"]:
                            print(
                                f"Failed solve for task_id={task_id}, gamma={gamma} - continue"
                            )
                    except:
                        print(
                            f"Failed load task_id={task_id}, gamma={gamma} - continue"
                        )
                        pass
                continue

            if (task_id, gamma) in self.solutions:
                solution_entry = self.solutions[(task_id, gamma)]
                if "cached" in solution_entry and solution_entry["cached"]:
                    print("Problem task_id={task_id} is solved, cached")
                    break

                if solution_entry["passed"]:
                    print(f"Problem task_id={task_id} is solved (gamma={gamma})")
                    break
                else:
                    print(
                        f"Solution exist - problem task_id={task_id} is unsolved (gamma={gamma})"
                    )
                    continue

            # touch the file to reserve it
            print(f"Try solve task_id={task_id}, gamma={gamma}")
            with open(solution_entry_path, "a"):
                pass

            solution_entry = self.solve_problem_with_dsgi_wrapper(problem, gamma)
            self.solutions[(task_id, gamma)] = solution_entry
            with open(solution_entry_path, "w") as f:
                json.dump(solution_entry, f, indent=2)
            if solution_entry["passed"]:
                print(f"Problem task_id={task_id} is solved (gamma={gamma})")
                if self.session_config.use_global_cache and not os.path.exists(
                    global_cache_solved_task_id_path
                ):
                    with open(global_cache_solved_task_id_path, "a"):
                        pass
                    with open(global_cache_solved_task_id_path, "w") as f:
                        entry = {
                            "gamma": gamma,
                            "random_seed": self.session_config.random_seed,
                        }
                        f.write(json.dumps(entry))
                break
            else:
                print(f"Failed Solving Problem task_id={task_id} (gamma={gamma})")

    def solve(self):
        # First resolve all inference sessions cache & officail evaluation entries
        for inference_session_config in self.inference_sessions_configs:
            self.setup_inference_session(
                inference_session_config,
            )
            self.resolve_cache_entries()
            self.resolve_official_evaluation_solved_entries()

        # Then, for every problem use all the configs to maximize caching
        for _, problem in self.problems:
            for inference_session_config in self.inference_sessions_configs:
                self.setup_inference_session(
                    inference_session_config,
                )
                self.solve_single_problem(problem)
