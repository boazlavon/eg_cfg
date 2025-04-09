MAX_NEW_TOKENS = 512

PROMPT_TYPE__DEEPSEEK_BASE = "deepseek_base"
PROMPT_TYPE__DEEPSEEK_INSTRUCT = "deepseek_instruct"
PROMPT_TYPE__INSTRUCT_LONG_CODE_PROMPT = "long_code"
PROMPT_TYPE__CUSTOM_PROMPT_SIMPLE = "custom_simple"

VALID_PROMPT_TYPES = [
    PROMPT_TYPE__DEEPSEEK_BASE,
    PROMPT_TYPE__DEEPSEEK_INSTRUCT,
    PROMPT_TYPE__INSTRUCT_LONG_CODE_PROMPT,
    # PROMPT_TYPE__CUSTOM_PROMPT_SIMPLE,
]

NEAREST_FUTURE_DYNAMIC_SIGNAL_PATTERN = """
# Function:
{function_code}

# Invocation:
{test_case}

# Execution Trace: {trace}
"""

SINGLE_DYNAMIC_SIGNAL_PATTERN = """
# Invocation:
{test_case}

# Execution Trace: {trace}
"""

NEAREST_FUTURE_DYNAMIC_SIGNAL_PROMPT = """
### Runtime Behavior for Candidate Continuations
Below are execution traces from running the response function after appending several possible future continuations. These continuations represent plausible ways the function might continue from its current state. They are not necessarily full solutions—some may be partial, exploratory, or incomplete.

For each candidate continuation, multiple test cases (invocations) were executed to observe its behavior under different inputs. Each entry includes:
- A candidate version of the function
- A specific test case used for invocation
- The resulting execution trace for that test case

These dynamic signals can help you better understand how different plausible continuations behave at runtime, and guide you toward a more accurate solution.

{dynamic_signals}
"""

PARTIAL_EXECUTION_DYNAMIC_SIGNAL_PROMPT = """
### Runtime Behavior up to the Last Valid Line
This trace reflects the actual runtime behavior of the response function executed up to the last syntactically or semantically valid line—before the function was completed. It captures how the current partial implementation behaves, which can provide useful context for continuing the function.

Typically, one or more test cases (invocations) are run against this partial version to observe any runtime behavior, including crashes, exceptions, or intermediate outputs.

Use this information to better understand how the partial function performs so far, and to guide the next steps in completing the function correctly.

{dynamic_signals}
"""

BACKWARD_DYNAMIC_SIGNAL_PROMPT = """
### Runtime Behavior of Invalid Solutions
The following examples show complete function solutions that failed to pass validation. These solutions were tested using assertions, and at least one assertion failed during execution—typically resulting in an AssertionError.

Each entry includes:
- A full function solution that failed at least one test
- A specific test case (assertion) that triggered the failure
- The resulting execution trace

Use this information to recognize and avoid common mistakes in incorrect solutions, and to guide your next attempt toward correct and robust behavior.

{dynamic_signals}
"""

BACKWARD_DYNAMIC_SIGNAL_PATTERN = """
# Invalid Solution:
{function_code}

# Test Case (Assertion):
{test_case}

# Execution Trace:
{trace}
"""

DYNAMIC_SIGNAL_PROMPT_REPLACE_STRING_INSTRUCT_BEGIN = "### Response:"
DYNAMIC_SIGNAL_PROMPT_REPLACE_STRING_BASE_BEGIN = "[BEGIN]"
DYNAMIC_SIGNAL_PROMPT_REPLACE_STRING_BASE_END = "[DONE]"
INSTRUCT_MODEL_PYTHON_CODE_START = "```python\n"
DYNAMIC_SIGNAL_PROMPT_INSTRUCT_MODEL_START_FUNCTION_MARKER = (
    DYNAMIC_SIGNAL_PROMPT_REPLACE_STRING_INSTRUCT_BEGIN
)
DYNAMIC_SIGNAL_PROMPT_BASE_MODEL_START_FUNCTION_MARKER = (
    DYNAMIC_SIGNAL_PROMPT_REPLACE_STRING_BASE_BEGIN
)

DYNAMIC_SIGNAL__PARTIAL_EXECUTION = "PartialExecution"
DYNAMIC_SIGNAL__NEAREST_FUTURE_EXECUTION = "NearestFutureExecution"
DYNAMIC_SIGNAL__BACKWARD = "Backward"

SUPPORTED_DYNAMIC_SIGNALS = (
    DYNAMIC_SIGNAL__PARTIAL_EXECUTION,
    DYNAMIC_SIGNAL__NEAREST_FUTURE_EXECUTION,
    DYNAMIC_SIGNAL__BACKWARD,
)

GAMMAS = (0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1)
FILENAME_TEMPLATE = "task_id={task_id}_gamma={gamma}.json"
FILENAME_TEMPLATE_BACKWARD_SIGNAL = (
    "task_id={task_id}_gamma={gamma}_b={backward_signals_iteration}.json"
)

TASK__CODE_GENERATION = "CodeGeneration"
SUPPORTED_TASKS = (TASK__CODE_GENERATION,)

DEEPSEEK_13B_INSTRUCT_BASELINE_PASSED_PATH = "/home/ai_center/ai_users/boazlavon/data/code/DeepSeek-Coder/Evaluation/MBPP/tmp/deepseek-ai_deepseek-coder-1.3b-instruct_time1743422160_bs1_shot_log_python.json.task_ids.json"
DEEPSEEK_13B_INSTRUCT_BASELINE_RESULTS_PATH = "/home/ai_center/ai_users/boazlavon/data/code/DeepSeek-Coder/Evaluation/MBPP/tmp/deepseek-ai_deepseek-coder-1.3b-instruct_time1743422160_bs1_shot_log_python.json"
DEEPSEEK_13B_INSTRUCT_MODEL_NAME = "deepseek-ai/deepseek-coder-1.3b-instruct"
OFFICIAL_PASSED_TASK_IDS_PATH = {
    DEEPSEEK_13B_INSTRUCT_MODEL_NAME: DEEPSEEK_13B_INSTRUCT_BASELINE_PASSED_PATH
}
OFFICIAL_RESULT_PATH = {
    DEEPSEEK_13B_INSTRUCT_MODEL_NAME: DEEPSEEK_13B_INSTRUCT_BASELINE_RESULTS_PATH
}

MBPP_SIZE = 500

GUIDANCE_STRATEGY__TOKEN_GUIDANCE = "token_guidance"
GUIDANCE_STRATEGY__PERSISTENT_PREFIX_GUIDANCE = "persistent_prefix_guidance"
GUIDANCE_STRATEGY__LINE_GUIDANCE = "line_guidance"
GUIDANCE_STRATEGIES = (
    GUIDANCE_STRATEGY__TOKEN_GUIDANCE,
    GUIDANCE_STRATEGY__PERSISTENT_PREFIX_GUIDANCE,
    GUIDANCE_STRATEGY__LINE_GUIDANCE,
)
