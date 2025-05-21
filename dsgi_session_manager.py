import random
import datetime
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
    extract_function_signature,
)
from model_utils import calculate_tokens_length
from mbpp_utils import parse_mbpp_assert_statement, load_official_results
from code_generation_utils import (
    generate_code_solutions,
    is_valid_python,
    raw_outputs_to_new_code,
)
from dsgi_injection_manager import DsgiInjectionManager
from model_utils import setup_device, load_model, load_tokenizer
from execution_manager import ExecutionManager
from args_utils import get_dynamic_signals_str
from probs_utils import stable_hash
from collections import defaultdict
from fw_utils import (
    inference_endpoint_dsgi,
    PostRequestTimeoutError,
    simple_query,
    complex_qwen_query,
    END_OF_CODE_STOP_SEQUENCE,
    START_OF_CODE_STOP_SEQUENCE,
)
from consts import *
from datetime import datetime


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


class StatisticsManager:
    def __init__(self):
        self.statistics = {}
        self.current_key = None

    def set_current_key(self, current_key):
        self.current_key = current_key

    def increate_counter(self, counter_key, count):
        if self.current_key not in self.statistics:
            self.statistics[self.current_key] = defaultdict(int)
        self.statistics[self.current_key][counter_key] += count

    def set_value(self, counter_key, value):
        if self.current_key not in self.statistics:
            self.statistics[self.current_key] = defaultdict(int)
        self.statistics[self.current_key][counter_key] = value


class DsgiSessionManager:
    def __init__(
        self,
        session_config,
        inference_sessions_configs,
    ):
        self.session_config = session_config
        self.inference_sessions_configs = inference_sessions_configs
        self.inference_session = None
        self.use_local_hf_model = (
            self.session_config.deployment_type == DEPLOYMENT_TYPE__LOCAL_HF_MODEL
        )
        self.use_inference_endpoint = (
            self.session_config.deployment_type == DEPLOYMENT_TYPE__INFERENCE_ENDPOINT
        )
        assert (
            self.use_local_hf_model ^ self.use_inference_endpoint
        ), "Exactly one of 'use_local_hf_model' or 'use_inference_endpoint' must be True"

    def validate_args(self):
        for inference_session_config in self.inference_sessions_configs:
            assert inference_session_config["prompt_type"] in (
                PROMPT_TYPE__DEEPSEEK_BASE,
                PROMPT_TYPE__DEEPSEEK_INSTRUCT,
                PROMPT_TYPE__INSTRUCT_LONG_CODE_PROMPT,
            )

    def setup(self):
        self.solutions = {}
        assert (
            self.session_config.model_name
            in SUPPORTED_MODELS_ON_DEPLOYMENTS[self.session_config.deployment_type]
        ), f"Model \"{self.session_config.model_name}\" is currently not supported for \"{self.session_config.deployment_type}\" deployment."
        if self.use_local_hf_model:
            self.device = setup_device()
            self.model, self.tokenizer = load_model(
                self.session_config.model_name, self.device
            )
        elif self.use_inference_endpoint:
            self.device = None
            self.model = None
            self.tokenizer = load_tokenizer(self.session_config.model_name)

        self.execution_manager = ExecutionManager(
            self.tokenizer,
            function_signature=None,
            minimal_trace=self.session_config.minimal_trace,
            debug_mode=self.session_config.debug_mode,
        )
        self.stats_manager = StatisticsManager()
        self.problems = load_mbpp_problems()
        self.problems = list(self.problems.items())
        if self.session_config.start_idx and self.session_config.end_idx:
            self.problems = self.problems[
                self.session_config.start_idx : self.session_config.end_idx
            ]
        if self.session_config.is_prod:
            random.shuffle(self.problems)

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
        os.makedirs(results_dir, exist_ok=True)
        solved_tasks_cache_dir = os.path.join(
            self.session_config.results_dir,
            "mbpp",
            self.session_config.model_name.replace("/", "_"),
            ".solved_tasks_cache",
        )
        if self.session_config.use_global_cache:
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

    def resolve_baseline_solved_entries(self):
        gamma = 0.0
        if self.session_config.model_name == DEEPSEEK_V3_0324_MODEL_NAME_HF:
            official_passed_task_ids = DEEPSEEK_V3_0324_SOLVED_TASK_IDS
        elif self.session_config.model_name == QWEN3_253B_MODEL_NAME_HF:
            official_passed_task_ids = QWEN3_SOLVED_TASK_IDS
        else:
            official_passed_task_ids, official_results = load_official_results(
                self.session_config.model_name
            )
        if not official_passed_task_ids:
            return
        if self.session_config.is_prod:
            random.shuffle(official_passed_task_ids)
        for _, problem in self.problems:
            task_id = problem["task_id"]
            if (
                not self.session_config.model_name
                in (DEEPSEEK_V3_0324_MODEL_NAME_HF, QWEN3_253B_MODEL_NAME_HF)
            ) and (not task_id in official_passed_task_ids):
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
                solution_entry["stats"] = {}
                solution_entry["retry"] = -1
                solution_entry["random_seed"] = -1
                self.solutions[(task_id, gamma)] = solution_entry

                print(solution_entry_path)
                with open(solution_entry_path, "w") as f:
                    json.dump(solution_entry, f, indent=2)

            elif self.session_config.model_name in (
                DEEPSEEK_V3_0324_MODEL_NAME_HF,
                QWEN3_253B_MODEL_NAME_HF,
            ):
                BASELINE_TRIALS_BASE = {
                    DEEPSEEK_V3_0324_MODEL_NAME_HF: "web_trials/baseline/mbpp/deepseek-ai_DeepSeek-V3-0324",
                    QWEN3_253B_MODEL_NAME_HF: "web_trials/baseline/mbpp/Qwen_Qwen3-235B-A22B",
                }
                BASELINE_DIRS = ["baseline_ln", "baseline_lci_ln"]
                baseline_trial_base = BASELINE_TRIALS_BASE[
                    self.session_config.model_name
                ]
                for baseline_dir in BASELINE_DIRS:
                    results_dir = os.path.join(baseline_trial_base, baseline_dir)
                    bs_solution_entry_path = get_solution_filepath(
                        results_dir,
                        task_id,
                        gamma=0.0,
                    )
                    if not os.path.exists(bs_solution_entry_path):
                        continue
                    with open(bs_solution_entry_path) as f:
                        bs_solution_entry = json.load(f)
                    for gamma in (0, 0.0):
                        solution_entry_path = get_solution_filepath(
                            self.inference_session.results_dir, task_id, gamma
                        )
                        with open(solution_entry_path, "w") as f:
                            json.dump(bs_solution_entry, f, indent=2)
                    if bs_solution_entry["passed"]:
                        break

    def build_dsgi_injection_manager_and_prompt(
        self, problem, gamma, function_signature=None
    ):
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
                "stats_manager": self.stats_manager,
                "use_local_hf_model": self.use_local_hf_model,
                "use_inference_endpoint": self.use_inference_endpoint,
                "model_name": self.session_config.model_name,
                "debug_mode": self.session_config.debug_mode
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
            debug_mode=self.session_config.debug_mode
        )
        return dsgi_injection_manager, prompt

    def solve_problem_with_dsgi(
        self,
        problem,
        gamma,
    ):
        solution = None
        early_stop = False
        function_signature = None
        dsgi_injection_manager, prompt = self.build_dsgi_injection_manager_and_prompt(
            problem, gamma, function_signature
        )
        initial_prompt_input_ids_len = calculate_tokens_length(self.tokenizer, prompt)
        stats_manager = dsgi_injection_manager.adapter.stats_manager
        if self.use_local_hf_model:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            outputs = generate_code_solutions(
                self.model,
                self.tokenizer,
                dsgi_injection_manager,
                inputs,
                num_return_sequences=1,
                prompt_type=self.inference_session.inference_session_config[
                    "prompt_type"
                ],
                stats_manager=stats_manager,
            )
        elif self.use_inference_endpoint:
            function_signature = extract_function_signature(problem["code"])
            # Instruct Model to write the final solution in the last code block
            # This models tend to use code block as part of their reasoning steps
            # and we need to way to decide where to start EG-CFG
            if self.session_config.model_name == QWEN3_253B_MODEL_NAME_HF:
                prompt = prompt.replace("Deepseek Coder", "Qwen3")
                prompt = prompt.replace("Deepseek", "Qwen")
                usage_comment = PYTHON_CODE_TAGS_USAGE_INSTRUCTION_QWEN

            if self.session_config.model_name == DEEPSEEK_V3_0324_MODEL_NAME_HF:
                usage_comment = PYTHON_CODE_TAGS_USAGE_INSTRUCTION_DS

            usage_comment_insertion_positions = [
                "Allowing **incremental execution and debugging**",  # long code prompt
                "Examples are listed as follows:",  # deepseek instruct prompt
            ]
            for insertion_position in usage_comment_insertion_positions:
                if insertion_position in prompt:
                    prompt = prompt.replace(
                        insertion_position, f"{insertion_position}\n{usage_comment}"
                    )
                    break

            if gamma > 0.0:
                outputs, early_stop = inference_endpoint_dsgi(
                    prompt,
                    self.tokenizer,
                    self.session_config.model_name,
                    dsgi_injection_manager,
                    function_signature,
                )
            else:  # gamma == 0
                # We resolve all gamma == 0.0. BUT it can be here in case we disable
                # resolve all gamma = 0.0 forehead
                # for some reason
                prompt_input_ids = self.tokenizer(prompt, return_tensors="pt")[
                    "input_ids"
                ]
                _, solution, completion_tokens = complex_qwen_query(
                    prompt,
                    function_signature,
                    self.session_config.model_name,
                    temperture=dsgi_injection_manager.adapter.temperature,
                    max_tokens=COMPLEX_QWEN_QUERY_MAX_TOKENS,
                    verbose=True,
                )
                if self.stats_manager is not None:
                    self.stats_manager.increate_counter(
                        "guidance_input_tokens", prompt_input_ids.shape[1]
                    )
                    self.stats_manager.increate_counter(
                        "guidance_output_tokens", completion_tokens
                    )
                assert solution
                return solution
            if early_stop:
                print("Early Stop detected!")
                solution = outputs
        if solution is None:
            new_codes = raw_outputs_to_new_code(
                outputs,
                self.tokenizer,
                initial_prompt_input_ids_len,
                self.inference_session.inference_session_config["prompt_type"],
                stats_manager=self.stats_manager,
            )
            solution = new_codes[0]
        return solution

    def solve_problem_with_dsgi_wrapper(self, problem, gamma):
        task_id = problem["task_id"]
        test_cases = problem["test_list"]
        self.stats_manager.set_current_key((task_id, gamma))
        start_time = datetime.now()
        start_time_str = start_time.strftime("%Y-%m-%d %H:%M:%S")
        if self.stats_manager is not None:
            self.stats_manager.set_value("start_time", start_time_str)
        print(f"[START] {start_time_str}")
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
                # We want to be able to use different configs is different
                # orders but make them agnostic to their order.
                # Each config has an IID seed 
                # transformed to range (40,40+999)
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
                )
                random_seed = stable_hash(iid_arg)
                random_seed = random_seed % 1000
                random_seed += self.session_config.random_seed
                random_seed += retry_idx
                random.seed(random_seed)
                torch.manual_seed(random_seed)
                torch.cuda.manual_seed_all(random_seed)

            try:
                solution = self.solve_problem_with_dsgi(problem, gamma)
                print(solution)
            except KeyboardInterrupt:
                exit(1)
            except PostRequestTimeoutError as e:
                general_error = str(type(e))
                tb = traceback.format_exc()
                print(tb)
                if not self.session_config.is_prod:
                    raise e
                else:
                    print("Avoid Server DDoS")
                    time.sleep(random.uniform(30, 60))
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

            end_time = datetime.now()
            duration = end_time - start_time
            end_time_str = end_time.strftime("%Y-%m-%d %H:%M:%S")
            if self.stats_manager is not None:
                self.stats_manager.set_value("end_time", end_time_str)
                self.stats_manager.set_value("duration", str(duration))
            print(f"[END] {end_time_str}")
            print(f"[DURATION] {duration}")
            solution_entry["stats"] = dict(
                self.stats_manager.statistics[(task_id, gamma)]
            )
            print(solution_entry["stats"])
            solution_entry["retry"] = retry_idx
            # solution_entry["random_seed"] = random_seed
            if solution_entry["passed"]:
                print(f"Problem task_id={task_id} is solved. (gamma={gamma})")
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
                    "passed": True,
                    "accuracy": 1.0,
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
        for inference_session_config in self.inference_sessions_configs:
            self.setup_inference_session(
                inference_session_config,
            )
            self.resolve_baseline_solved_entries()

        for _, problem in self.problems:
            for inference_session_config in self.inference_sessions_configs:
                self.setup_inference_session(
                    inference_session_config,
                )
                self.solve_single_problem(problem)
