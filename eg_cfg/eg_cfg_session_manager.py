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
from datasets_utils import (
    format_task_prompt,
    load_mbpp_problems,
    load_humaneval_problems,
    load_codecontests_problems,
    extract_function_signature,
)
from eval_utils import (
    run_tests,
)
from exec_eval_utils import exec_eval__run_tests, ExecEval__APICommunication
from model_utils import calculate_tokens_length
from datasets_utils import parse_assert_statement
from code_generation_utils import (
    generate_code_solutions,
    raw_outputs_to_new_code,
)
from eg_cfg_injection_manager import EgCfgInjectionManager
from model_utils import setup_device, load_model, load_tokenizer
from execution_manager import ExecutionManager
from args_utils import get_dynamic_signals_str
from probs_utils import stable_hash
from collections import defaultdict
from inference_endpoint_utils import (
    inference_endpoint_eg_cfg,
    inference_endpoint_eg_cfg_gamma_1_optimization,
    PostRequestTimeoutError,
    reasoning_tokens_query,
)
from consts import *
from datetime import datetime

EXEC_EVAL_API_COMM = None


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


class EgCfgSessionManager:
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
        ), f'Model "{self.session_config.model_name}" is currently not supported for "{self.session_config.deployment_type}" deployment.'
        if self.session_config.deployment_type == DEPLOYMENT_TYPE__INFERENCE_ENDPOINT:
            assert (
                self.session_config.inference_endpoint_api_key
            ), "Missing inference endpoint key."
            assert (
                self.session_config.inference_endpoint_url
            ), "Missing inference endpoint URL."
            os.environ["FW_KEY"] = self.session_config.inference_endpoint_api_key
            os.environ["FW_ENDPOINT_URL"] = self.session_config.inference_endpoint_url

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
            debug=self.session_config.debug_mode,
        )
        self.stats_manager = StatisticsManager()
        assert self.session_config.dataset in AVAILABLE_DATASETS
        if self.session_config.dataset == DATASET__MBPP:
            self.problems = load_mbpp_problems()  # uses test_list
            self.eval_dataset = (
                self.problems
            )  # uses eval_test_list in humaneval, test_list in mbpp
        if self.session_config.dataset == DATASET__HUMANEVAL:
            self.problems = load_humaneval_problems()
            self.eval_dataset = (
                self.problems
            )  # uses eval_test_list in humaneval, test_list in mbpp
        if self.session_config.dataset == DATASET__CODECONTESTS:
            self.problems = load_codecontests_problems()
            self.eval_dataset = self.problems

        self.problems = list(self.problems.items())
        if self.session_config.start_idx and self.session_config.end_idx:
            self.problems = self.problems[
                self.session_config.start_idx : self.session_config.end_idx
            ]
        if self.session_config.is_prod:
            random.shuffle(self.problems)
        if self.session_config.results_dir:
            self.session_config.results_dir = os.path.join(
                os.getcwd(), self.session_config.results_dir
            )
            os.makedirs(self.session_config.results_dir, exist_ok=True)
        if self.session_config.exec_eval:
            global EXEC_EVAL_API_COMM
            if EXEC_EVAL_API_COMM is None:
                EXEC_EVAL_API_COMM = ExecEval__APICommunication(
                    self.session_config.exec_eval_host_ip,
                    self.session_config.exec_eval_host_port,
                )

    def create_results_dir(self, session_config, inference_session_config):
        dynamic_signals_str = get_dynamic_signals_str(inference_session_config)
        dataset_name = self.session_config.dataset
        if self.session_config.exec_eval:
            dataset_name = f"{dataset_name}__ExecEval"
        results_dir = os.path.join(
            self.session_config.results_dir,
            dataset_name,
            session_config.model_name.replace("/", "_"),
            dynamic_signals_str,
        )
        return results_dir

    def setup_inference_session(self, inference_session_config):
        results_dir = self.create_results_dir(
            self.session_config, inference_session_config
        )
        os.makedirs(results_dir, exist_ok=True)
        dataset_name = self.session_config.dataset
        if self.session_config.exec_eval:
            dataset_name = f"{dataset_name}__ExecEval"
        solved_tasks_cache_dir = os.path.join(
            self.session_config.results_dir,
            dataset_name,
            self.session_config.model_name.replace("/", "_"),
            SOLVED_TASKS_CACHE_DIRNAME,
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

    def get_solution_filepath(self, results_dir, task_id, gamma):
        if self.session_config.dataset == DATASET__HUMANEVAL:
            task_id = task_id.replace("/", "_")
        filename = FILENAME_TEMPLATE.format(task_id=task_id, gamma=gamma)
        return os.path.join(results_dir, filename)

    def build_eg_cfg_injection_manager_and_prompt(
        self, problem, gamma, function_signature=None
    ):
        test_cases_to_prompt = problem["test_list"]
        use_eg_cfg = True
        use_detector = True
        if gamma == 0 and self.session_config.dataset == DATASET__HUMANEVAL:
            prompt = problem["prompt"]
            end_string = CODE_BORDER_TOKEN
        elif (
            self.inference_session.inference_session_config["prompt_type"]
            == PROMPT_TYPE__INSTRUCT_LONG_CODE_PROMPT
        ):
            prompt, _ = format_task_prompt(problem, False)
            end_string = CODE_BORDER_TOKEN

        elif self.inference_session.inference_session_config["prompt_type"] in (
            PROMPT_TYPE__DEEPSEEK_BASE,
            PROMPT_TYPE__DEEPSEEK_INSTRUCT,
        ):
            if self.session_config.dataset == DATASET__MBPP:
                if (
                    self.inference_session.inference_session_config["prompt_type"]
                    == PROMPT_TYPE__DEEPSEEK_BASE
                ):
                    prompts_path = os.path.join(
                        MAIN_DATA_DIR,
                        DEEPSEEK_PROMPT_DIRNAME,
                        MBPP_BASE_PROMPT_FILENAME,
                    )
                    end_string = DYNAMIC_SIGNAL_PROMPT_REPLACE_STRING_BASE_END
                if (
                    self.inference_session.inference_session_config["prompt_type"]
                    == PROMPT_TYPE__DEEPSEEK_INSTRUCT
                ):
                    prompts_path = os.path.join(
                        MAIN_DATA_DIR,
                        DEEPSEEK_PROMPT_DIRNAME,
                        MBPP_INSTRUCT_PROMPT_FILENAME,
                    )
                    end_string = CODE_BORDER_TOKEN
                with open(prompts_path, "r") as f:
                    prompts = json.load(f)
                    prompt = prompts[str(problem["task_id"])]
            elif self.session_config.dataset in (
                DATASET__HUMANEVAL,
                DATASET__CODECONTESTS,
            ):
                prompt, _ = format_task_prompt(problem, True)
                end_string = CODE_BORDER_TOKEN

        print(self.inference_session.inference_session_config)
        dynamic_signals_types = [
            dynamic_signal_type
            for dynamic_signal_type in SUPPORTED_DYNAMIC_SIGNALS
            if self.inference_session.inference_session_config[
                dynamic_signal_type
            ].is_enabled
        ]
        if use_eg_cfg:
            eg_cfg_injection_manager = None
            task_kwargs = {
                "model": self.model,
                "tokenizer": self.tokenizer,
                "device": self.device,
                "function_signature": function_signature,
                "test_cases": test_cases_to_prompt,
                "initial_prompt": prompt,
                "dynamic_signals_types": dynamic_signals_types,
                "bs_candidates_count": self.inference_session.inference_session_config[
                    DYNAMIC_SIGNAL__MULTIPLE_CANDIDATES_EXECUTION
                ].bs_candidates_count,
                "bs_new_signal_threshold": self.inference_session.inference_session_config[
                    DYNAMIC_SIGNAL__MULTIPLE_CANDIDATES_EXECUTION
                ].bs_new_signal_threshold,
                "temperature": self.inference_session.inference_session_config[
                    DYNAMIC_SIGNAL__MULTIPLE_CANDIDATES_EXECUTION
                ].temperature,
                "bs_completion_horizon": self.inference_session.inference_session_config[
                    DYNAMIC_SIGNAL__MULTIPLE_CANDIDATES_EXECUTION
                ].bs_completion_horizon,
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
                "execute_io": self.session_config.dataset == DATASET__CODECONTESTS,
                # "debug_mode": self.session_config.debug_mode,
            }

            detector_kwargs = {}
            if use_detector:
                initial_prompt_input_ids_len = calculate_tokens_length(
                    self.tokenizer, prompt
                )
                if problem.get("entry_point"):
                    function_name = problem.get("entry_point")
                else:
                    function_name, _, _ = parse_assert_statement(
                        test_cases_to_prompt[0]
                    )
                detector_kwargs = {
                    "tokenizer": self.tokenizer,
                    "initial_prompt_input_ids_len": initial_prompt_input_ids_len,
                    "function_name": function_name,
                    "end_string": end_string,
                }
        task = TASK__CODE_GENERATION
        eg_cfg_injection_manager = EgCfgInjectionManager(
            self.tokenizer,
            task,
            task_kwargs,
            gamma,
            detector_kwargs,
            use_detector=use_detector,
            top_probs_count=self.session_config.top_probs,
            debug_mode=self.session_config.debug_mode,
        )
        return eg_cfg_injection_manager, prompt

    def solve_problem_with_eg_cfg(
        self,
        problem,
        gamma,
    ):
        solution = None
        early_stop = False
        function_signature = None
        eg_cfg_injection_manager, prompt = (
            self.build_eg_cfg_injection_manager_and_prompt(
                problem, gamma, function_signature
            )
        )
        initial_prompt_input_ids_len = calculate_tokens_length(self.tokenizer, prompt)
        stats_manager = eg_cfg_injection_manager.adapter.stats_manager
        if self.use_local_hf_model:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            outputs = generate_code_solutions(
                self.model,
                self.tokenizer,
                eg_cfg_injection_manager,
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

            max_tokens = MAX_NEW_TOKENS
            if self.session_config.dataset == DATASET__CODECONTESTS:
                max_tokens *= 3

            if gamma > 0.0:
                if (
                    gamma == GAMMA_1_OPTIMIZATION_VALUE
                ):  # gamma == 1.0001 is a special case for optimization for gamma=1
                    outputs, early_stop, inference_initial_prompt_input_ids_len = (
                        inference_endpoint_eg_cfg_gamma_1_optimization(
                            prompt,
                            self.tokenizer,
                            self.session_config.model_name,
                            eg_cfg_injection_manager,
                            function_signature,
                            function_name=problem.get("entry_point"),
                            max_tokens=max_tokens,
                        )
                    )
                else:
                    outputs, early_stop, inference_initial_prompt_input_ids_len = (
                        inference_endpoint_eg_cfg(
                            prompt,
                            self.tokenizer,
                            self.session_config.model_name,
                            eg_cfg_injection_manager,
                            function_signature,
                            function_name=problem.get("entry_point"),
                            max_tokens=max_tokens,
                        )
                    )
                if inference_initial_prompt_input_ids_len is not None:
                    initial_prompt_input_ids_len = (
                        inference_initial_prompt_input_ids_len
                    )
            else:  # gamma == 0
                prompt_input_ids = self.tokenizer(prompt, return_tensors="pt")[
                    "input_ids"
                ]
                _, solution, completion_tokens = reasoning_tokens_query(
                    prompt,
                    function_signature,
                    self.session_config.model_name,
                    temperture=eg_cfg_injection_manager.adapter.temperature,
                    max_tokens=REASONING_TOKENS_QUERY_MAX_TOKENS,
                    verbose=True,
                    function_name=problem.get("entry_point"),
                    return_raw=(self.session_config.dataset == DATASET__HUMANEVAL),
                )
                if self.session_config.dataset == DATASET__HUMANEVAL:
                    solution = f"{prompt}\n{solution}"
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

    def solve_problem_with_eg_cfg_wrapper(self, problem, gamma):
        task_id = problem["task_id"]
        eval_problem = self.eval_dataset[task_id]
        if self.session_config.dataset in (DATASET__MBPP,):
            test_cases_to_eval = eval_problem["test_list"]
        if self.session_config.dataset in (DATASET__HUMANEVAL, DATASET__CODECONTESTS):
            test_cases_to_eval = eval_problem["eval_test_list"]
        self.stats_manager.set_current_key((task_id, gamma))
        start_time = datetime.now()
        start_time_str = start_time.strftime("%Y-%m-%d %H:%M:%S")
        if self.stats_manager is not None:
            self.stats_manager.set_value("start_time", start_time_str)
        random_seed = 0
        print(f"[START] {start_time_str}")
        for retry_idx in range(self.session_config.retries_count):
            general_error = None
            tb = None
            # no need to solve gamma = 0 multiple times
            if gamma == 0 and retry_idx > 0:
                break
            print(f"Retry #{retry_idx + 1}")
            if self.inference_session.inference_session_config[
                DYNAMIC_SIGNAL__MULTIPLE_CANDIDATES_EXECUTION
            ].is_enabled:
                # We want to be able to use different configs is different
                # orders but make them agnostic to their order.
                # Each config has an IID seed
                # transformed to range (random_seed, random_seed + 999)
                iid_arg = (
                    self.inference_session.inference_session_config[
                        DYNAMIC_SIGNAL__MULTIPLE_CANDIDATES_EXECUTION
                    ].temperature,
                    self.inference_session.inference_session_config[
                        DYNAMIC_SIGNAL__MULTIPLE_CANDIDATES_EXECUTION
                    ].bs_candidates_count,
                    self.inference_session.inference_session_config[
                        DYNAMIC_SIGNAL__MULTIPLE_CANDIDATES_EXECUTION
                    ].bs_completion_horizon,
                    self.inference_session.inference_session_config[
                        DYNAMIC_SIGNAL__PARTIAL_EXECUTION
                    ].is_enabled,
                )
                random_seed = stable_hash(iid_arg)
                random_seed = random_seed % RANDOM_SEED_RANGE_SIZE
                random_seed += self.session_config.random_seed
                random_seed += retry_idx
                random.seed(random_seed)
                torch.manual_seed(random_seed)
                torch.cuda.manual_seed_all(random_seed)

            try:
                solution = self.solve_problem_with_eg_cfg(problem, gamma)
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

            end_time = datetime.now()

            tests_start_time = datetime.now()
            tests_start_time_str = tests_start_time.strftime("%Y-%m-%d %H:%M:%S")
            if self.stats_manager is not None:
                self.stats_manager.set_value("tests_start_time", tests_start_time_str)

            io_flag = self.session_config.dataset == DATASET__CODECONTESTS
            if self.session_config.exec_eval:
                global EXEC_EVAL_API_COMM
                solution_entry = exec_eval__run_tests(
                    solution, test_cases_to_eval, EXEC_EVAL_API_COMM
                )
            else:
                solution_results = run_tests(solution, test_cases_to_eval, io_flag)
                solution_entry = format_results(
                    solution, solution_results, general_error, tb
                )

            tests_end_time = datetime.now()
            tests_end_time_str = tests_end_time.strftime("%Y-%m-%d %H:%M:%S")
            if self.stats_manager is not None:
                self.stats_manager.set_value("tests_end_time", tests_end_time_str)

            duration = end_time - start_time
            tests_duration = tests_end_time - tests_start_time
            end_time_str = end_time.strftime("%Y-%m-%d %H:%M:%S")
            if self.stats_manager is not None:
                self.stats_manager.set_value("end_time", end_time_str)
                self.stats_manager.set_value("duration", str(duration))
                self.stats_manager.set_value("tests_duration", str(tests_duration))
            print(f"[END] {end_time_str}")
            print(f"[DURATION] {duration}")
            solution_entry["stats"] = dict(
                self.stats_manager.statistics[(task_id, gamma)]
            )
            print(solution_entry["stats"])
            solution_entry["retry"] = retry_idx
            solution_entry["random_seed"] = random_seed
            if solution_entry["passed"]:
                print(f"Problem task_id={task_id} is solved. (gamma={gamma})")
                break

        return solution_entry

    def solve_single_problem(self, problem):
        task_id = problem["task_id"]
        print(f"task_id: {task_id}")
        # pprint.pprint(problem)
        print()

        global_cache_solved_task_id_path = os.path.join(
            self.inference_session.solved_tasks_cache_dir, f"{task_id}.json"
        )
        for gamma in self.session_config.gammas:
            print(f"task_id={task_id}, gamma={gamma}")
            solution_entry_path = self.get_solution_filepath(
                self.inference_session.results_dir, task_id, gamma
            )

            if (
                self.session_config.use_global_cache
                and os.path.exists(global_cache_solved_task_id_path)
                and not os.path.exists(solution_entry_path)
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
                            solution_entry_dump = json.dumps(solution_entry)
                            if (
                                self.session_config.use_global_cache
                                and not os.path.exists(global_cache_solved_task_id_path)
                                and not solution_entry.get("global_cached")
                                and not "global_cached" in solution_entry_dump
                            ):
                                print(f"Added new global cache entry: {task_id}")
                                with open(global_cache_solved_task_id_path, "w") as f2:
                                    json.dump(solution_entry, f2, indent=2)
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

            solution_entry = self.solve_problem_with_eg_cfg_wrapper(problem, gamma)
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
                    with open(global_cache_solved_task_id_path, "w") as f2:
                        json.dump(solution_entry, f2, indent=2)
                break
            else:
                print(f"Failed Solving Problem task_id={task_id} (gamma={gamma})")

    def solve(self):
        for _, problem in self.problems:
            for inference_session_config in self.inference_sessions_configs:
                self.setup_inference_session(
                    inference_session_config,
                )
                self.solve_single_problem(problem)
