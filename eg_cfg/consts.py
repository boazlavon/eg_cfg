MAX_NEW_TOKENS = 512

PROMPT_TYPE__DEEPSEEK_BASE = "deepseek_base"
PROMPT_TYPE__DEEPSEEK_INSTRUCT = "deepseek_instruct"
PROMPT_TYPE__INSTRUCT_LONG_CODE_PROMPT = "long_code"
BASELINE_LLM_SAMPLING_TEMPERATURE = 0.7
DATASET__CODECONTESTS__HF_PATH = "deepmind/code_contests"
DATASET__MBPP__HF_PATH = "google-research-datasets/mbpp"
DATASET__MBPP_ET_HF_PATH = "dz1/CodeScore-MBPP-ET"
DATASET__HUMANEVAL_ET_HF_PATH = "dz1/CodeScore-HumanEval-ET"

EVALUATE_SOLUTION_IO_TIMEOUT_SEC = 15
EVALUATE_SOLUTION_TIMEOUT_SEC = 10
DEFAULT_MAX_WORKERS = 8

EXEC_EVAL_DEFAULT_HOST_PORT = 5000
EXEC_EVAL__TIMEOUT_SEC = 120
EVAL_TYPE__EG_CFG = "eval-eg-cfg"
EVAL_TYPE__EXEC_EVAL = "ExecEval"
AVAILABLE_EVAL_TYPES = (EVAL_TYPE__EG_CFG, EVAL_TYPE__EXEC_EVAL)

DATASET__MBPP = "mbpp"
DATASET__MBPP_ET = "mbpp-et"
DATASET__HUMANEVAL = "humaneval"
DATASET__HUMANEVAL_ET = "humaneval-et"
DATASET__CODECONTESTS = "CodeContests"
AVAILABLE_DATASETS = (DATASET__MBPP, DATASET__HUMANEVAL, DATASET__CODECONTESTS)
AVAILABLE_EVAL_DATASETS = (
    DATASET__MBPP_ET,
    DATASET__HUMANEVAL_ET,
    DATASET__CODECONTESTS,
)
ALL_DATASETS = (
    DATASET__MBPP,
    DATASET__HUMANEVAL,
    DATASET__CODECONTESTS,
    DATASET__MBPP_ET,
    DATASET__HUMANEVAL_ET,
)

EVAL_DEFAULT_WORKERS = 12  # Default number of workers per trial
EXEC_EVAL__LANGUAGE__PYTHON3 = "Python 3"
EXEC_EVAL__LANGUAGE__PYTHON3__LIMITS = {"nofile": 4}
EXEC_EVAL__EXECUTE_CODE_URL_TEMPLATE = (
    "http://{exec_eval_host_ip}:{exec_eval_host_port}/api/execute_code"
)
EXEC_EVAL__REQUEST_HEADERS = {"Content-Type": "application/json"}

CODE_CONTESTS__INVOCATION = "solve()"

# When just executed without any debugger (like in eval time)
# its ok to just override stdin and stdout
INJECT_IO_EVAL = """
import sys, io
s = {expected_stdin!r}
sys.stdin = io.StringIO(s)
sys.stdout = open("{stdout_path}", "w")
"""

GAMMA_1_OPTIMIZATION_VALUE = 1

# We cannot override sys.stdin in anyway because the debugger and
# the program share thesame stdin so it overrides the debugger stdin as well
# in this way we cannot execute the debugger commands in "traces_dumper/runner.py"
INJECT_IO_INSIDE_DEBUGGER = """
import sys, io

def input__custom(prompt=None):
    if prompt:
        print(prompt, end='', flush=True)

    if not hasattr(input__custom, "_gen"):
        input_string = {expected_stdin!r}  # Injected input string
        input__custom._gen = (line for line in input_string.splitlines())

    try:
        return next(input__custom._gen)
    except StopIteration:
        raise EOFError

def read__custom():
    return {expected_stdin!r}

def readline__custom():
    return input__custom() + '\\n'

def readlines__custom():
    return [line + '\\n' for line in {expected_stdin!r}.splitlines()]

def print__custom(*args, **kwargs):
    with open({stdout_path!r}, 'a') as f:
        kwargs_copy = kwargs.copy()
        sep = kwargs_copy.pop('sep', ' ')
        end = kwargs_copy.pop('end', '\\n')
        text = sep.join(str(arg) for arg in args) + end
        f.write(text)
"""

TESTS_COUNT_THRESHOLD = 3
EXAMPLE_CODE = """""
def solve():
    print('Hello')
"""

HUMANEVAL_INSTRUCTION_TEMPLATE = """
Write a function that performs the following task: {instruction}
It should have the following function signature: {function_signature}
"""

VALID_PROMPT_TYPES = [
    PROMPT_TYPE__DEEPSEEK_INSTRUCT,
    PROMPT_TYPE__INSTRUCT_LONG_CODE_PROMPT,
]

MULTIPLE_CANDIDATES_DYNAMIC_SIGNAL_PATTERN = """
# Function:
{function_code}

# Invocation:
{test_case}

# Execution Trace:\n{trace}
"""

SINGLE_DYNAMIC_SIGNAL_PATTERN = """
# Invocation:
{test_case}

# Execution Trace: {trace}
"""

MULTIPLE_CANDIDATES_DYNAMIC_SIGNAL_PROMPT = """
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

DYNAMIC_SIGNAL_PROMPT_REPLACE_STRING_INSTRUCT_BEGIN = "### Response:"
DYNAMIC_SIGNAL_PROMPT_REPLACE_STRING_BASE_BEGIN = "[BEGIN]"
DYNAMIC_SIGNAL_PROMPT_REPLACE_STRING_BASE_END = "[DONE]"
INSTRUCT_MODEL_PYTHON_CODE_START = "```python\n"
INSTRUCT_MODEL_PYTHON_CODE_START_TOK = "```python"
CODE_BORDER_TOKEN = "```"
END_OF_CODE_STOP_SEQUENCE = f"{CODE_BORDER_TOKEN}\n"
END_OF_SENTENCE_TOKEN = "<__end_of_sentence__>"
END_OF_TEXT_TOKEN = "<endoftext>"
COMPLEX_QUERY_STOP_CONDITION = (END_OF_TEXT_TOKEN, "<im_end>", END_OF_SENTENCE_TOKEN)

RANDOM_SEED_RANGE_SIZE = 1000
DYNAMIC_SIGNAL_PROMPT_INSTRUCT_MODEL_START_FUNCTION_MARKER = (
    DYNAMIC_SIGNAL_PROMPT_REPLACE_STRING_INSTRUCT_BEGIN
)
DYNAMIC_SIGNAL_PROMPT_BASE_MODEL_START_FUNCTION_MARKER = (
    DYNAMIC_SIGNAL_PROMPT_REPLACE_STRING_BASE_BEGIN
)

DYNAMIC_SIGNAL__PARTIAL_EXECUTION = "PartialExecution"
DYNAMIC_SIGNAL__MULTIPLE_CANDIDATES_EXECUTION = "MultipleCandidatesExecution"

SUPPORTED_DYNAMIC_SIGNALS = (
    DYNAMIC_SIGNAL__PARTIAL_EXECUTION,
    DYNAMIC_SIGNAL__MULTIPLE_CANDIDATES_EXECUTION,
)

CMDLINE_ARGS_ONLY_GAMMAS = [0.0, 0.5, 1, 3]
FILENAME_TEMPLATE = "task_id={task_id}_gamma={gamma}.json"

TASK__CODE_GENERATION = "CodeGeneration"
SUPPORTED_TASKS = (TASK__CODE_GENERATION,)
MBPP_INSTRUCT_PROMPT_FILENAME = "mbpp_instruct_prompts.json"
MBPP_BASE_PROMPT_FILENAME = "mbpp_base_prompts.json"
MAIN_DATA_DIR = "data"
DEEPSEEK_PROMPT_DIRNAME = "deepseek_mbpp_prompts"
DEEPSEEK_13B_INSTRUCT_MODEL_NAME = "deepseek-ai/deepseek-coder-1.3b-instruct"

DEEPSEEK_CODER_V2_LITE_INSTRUCT_MODEL_NAME = (
    "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
)

EXECUTION_TIMEOUT_SEC = 40
MBPP_SIZE = 500

SOLVED_TASKS_CACHE_DIRNAME = "solved_tasks"

GUIDANCE_STRATEGY__TOKEN_GUIDANCE = "token_guidance"
GUIDANCE_STRATEGY__PERSISTENT_PREFIX_GUIDANCE = "persistent_prefix_guidance"
GUIDANCE_STRATEGY__LINE_GUIDANCE = "line_guidance"
GUIDANCE_STRATEGIES = (
    GUIDANCE_STRATEGY__TOKEN_GUIDANCE,
    GUIDANCE_STRATEGY__LINE_GUIDANCE,
)
EARLY_STOP_THRESHOLD = 4
## FW Inference API
FW__MIN_BATCH_SIZE = 5
FW_UTILS__DEFAULT_TOP_P = 0.95
LOGPROBS_COUNT = 5
HTTP_REQUEST_TO_LLM_RETRIES_COUNT = 5
REQUEST_TIMEOUT_SEC = 30
QWEN_REQUEST_TIMEOUT_SEC = REQUEST_TIMEOUT_SEC * 3
MATCH_RETRIES_COUNT = 10

DEEPSEEK_V3_0324_MODEL_NAME_HF = "deepseek-ai/DeepSeek-V3-0324"
DEEPSEEK_V3_0324_MODEL_NAME_FW = "accounts/fireworks/models/deepseek-v3-0324"
QWEN3_253B_MODEL_NAME_HF = "Qwen/Qwen3-235B-A22B"
QWEN3_253B_MODEL_NAME_FW = "accounts/fireworks/models/qwen3-235b-a22b"

PSEUDO_BEAM_SEARCH_MAX_TOKENS = MAX_NEW_TOKENS
REASONING_TOKENS_QUERY_MAX_TOKENS = 2048

PSEUDO_BEAM_SEARCH_MAX_TOTAL_REQUESTS = 2
HF_MODEL_TO_FW_MODEL = {
    DEEPSEEK_V3_0324_MODEL_NAME_HF: DEEPSEEK_V3_0324_MODEL_NAME_FW,
    QWEN3_253B_MODEL_NAME_HF: QWEN3_253B_MODEL_NAME_FW,
}
HTTP_SUCCESS_CODE = 200
PYTHON_CODE_TAGS_USAGE_INSTRUCTION_QWEN = (
    "# IMPORTANT: All generated Python code MUST be enclosed EXCLUSIVELY within "
    "```python and ``` tags. No other formatting is acceptable.\n"
    "# The FINAL answer MUST be the LAST code block in the output, written with NO comments or text after it. "
    "It must be immediately followed by the <endoftext> token."
)
PYTHON_CODE_TAGS_USAGE_INSTRUCTION_DS = (
    "# IMPORTANT: All generated Python code MUST be enclosed EXCLUSIVELY within "
    "```python and ``` tags. No other formatting is acceptable.\n"
    "# The FINAL answer MUST be the LAST code block in the output, written with NO comments or text after it. "
    "It must be immediately followed by the <__end_of_sentence__> token."
)

SOLUTION_FUNCTION_STRUCTURE_INSTRUCTION = """
Implement your entire solution inside a function named `solve()`.

Strict requirements:
- Define the function exactly like this:

  def solve():

- The `solve()` function must NOT take any arguments or return anything.
- Use `input()` or `sys.stdin` to read input.
- Use `print()` to produce output.
- Do NOT include any code outside the `solve()` function.
- Do NOT call `solve()` yourself.

Example:
```python
def solve():
    name = input()
    print("Hello", name)
```

Additional implementation constraints:

- Do not define another function named solve inside the outer solve() function.
- Do not define solve() with any parameters (e.g., def solve(_, __): is invalid).
- Avoid wrapping the entire logic inside an inner solve() or another local function.
- Your entire solution must be implemented directly inside a single solve() function body.
- Avoid "flattening" patterns where you define internal functions that shadow or re-declare solve()—these will be rejected.

Violating this rule will result in rejection of the solution.
"""

DEPLOYMENT_TYPE__INFERENCE_ENDPOINT = "inference_endpoint"
DEPLOYMENT_TYPE__LOCAL_HF_MODEL = "local"
SUPPORTED_DEPLOYMENT_TYPES = (
    DEPLOYMENT_TYPE__INFERENCE_ENDPOINT,
    DEPLOYMENT_TYPE__LOCAL_HF_MODEL,
)
SUPPORTED_MODELS_ON_DEPLOYMENTS = {
    DEPLOYMENT_TYPE__INFERENCE_ENDPOINT: (
        DEEPSEEK_V3_0324_MODEL_NAME_HF,
        QWEN3_253B_MODEL_NAME_HF,
    ),
    DEPLOYMENT_TYPE__LOCAL_HF_MODEL: (DEEPSEEK_13B_INSTRUCT_MODEL_NAME,),
}

SESSION_CONFIGS_DEFAULT_VALUES = {
    "retries_count": 1,
    "use_global_cache": False,
    "minimal_trace": False,
    "exec_eval": False,
    "exec_eval_host_ip": None,
    "exec_eval_host_port": None,
    "top_probs": 0,
    "debug_mode": False,
    "start_idx": None,
    "end_idx": None,
    "random_seed": 40,
    "inference_endpoint_api_key": None,
    "inference_endpoint_url": None,
}

TIMEOUT_DELTA_MIN = 40

TEST_CASES_INSTRUCTION = """
Write a Python function that satisfies the following test cases:
>>> Test Cases:
{test_cases}
"""

LONG_CODE_INSTRUCTION_TEXT = """\
You are an AI programming assistant, utilizing the Deepseek Coder model, developed by Deepseek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer
### Instruction:
{problem_text}

Write a Python function that satisfies the following test cases:
>>> Test Cases:
{test_cases}

Your solution should be written in as many lines as possible.
This ensures that prefixes of your function remain valid Python programs.
Allowing **incremental execution and debugging**.

Write the function **step by step**, progressively introducing variables and logic.
Avoid using list comprehensions, lambda functions, or overly compact one-liners.
Instead, follow these guidelines:**

Avoid list comprehensions, use loops instead:
Incorrect:
```python
def square_numbers(lst):
    return [x ** 2 for x in lst]
```

Correct:
```python
def square_numbers(lst):
    squares = []
    for num in lst:
        squared_value = num ** 2
        squares.append(squared_value)
    return squares
```

Avoid inline expressions, use variables instead
Incorrect:
```python
def calculate_area(length, width):
    return (length * width) / 2
```

Correct:
```python
def calculate_area(length, width):
    product = length * width
    area = product / 2
    return area
```

Incorrect:
```python
result.append(x + y)
```

Correct:
```python
z = x + y
result.append(z)
```

Incorrect:
```python
def compute_value(a, b, c):
    return (a + b) * (c / (a - b) + (a * c) / (b + c))
```

Correct:
```python
def compute_value(a, b, c):
    term1 = a + b 
    term2 = a - b 
    term3 = c / term2 
    term4 = a * c / (b + c)
    result = term1 * (term3 + term4)
    return result
```

### Response:
"""

DEEPSEEK_INSTRUCT_TESTCASES_INSTRUCTION_TMP = """
>>>> Test Cases:
{test_cases}
"""

DEEPSEEK_INSTRUCT_TESTCASES_INSTRUCTION = """
>>> Test Cases:
{test_cases}
"""

DEEPSEEK_INSTRUCT_TEMPLATE = """\
You are an AI programming assistant, utilizing the Deepseek Coder model, developed by Deepseek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer
### Instruction:
Please refer the given examples and generate a python function for my problem.
Examples are listed as follows:

- Example 1:
>>> Problem:
Write a function to find the similar elements from the given two tuple lists.
>>> Test Cases:
assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)
assert similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4)
assert similar_elements((11, 12, 14, 13),(17, 15, 14, 13)) == (13, 14)

>>> Code:
```python
def similar_elements(test_tup1, test_tup2):
  res = tuple(set(test_tup1) & set(test_tup2))
  return (res)
```

- Example 2:
>>> Problem:
Write a python function to identify non-prime numbers.
>>> Test Cases:
assert is_not_prime(2) == False
assert is_not_prime(10) == True
assert is_not_prime(35) == True

>>> Code:
```python
import math
def is_not_prime(n):
    result = False
    for i in range(2,int(math.sqrt(n)) + 1):
        if n % i == 0:
            result = True
    return result
```

- Example 3:
>>> Problem:
Write a function to find the largest integers from a given list of numbers using heap queue algorithm.
>>> Test Cases:
assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],3)==[85, 75, 65]
assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],2)==[85, 75]
assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],5)==[85, 75, 65, 58, 35]

>>> Code:
```python
import heapq as hq
def heap_queue_largest(nums,n):
  largest_nums = hq.nlargest(n, nums)
  return largest_nums
```

Here is my problem:
>>> Problem:
{problem_text}
>>>> Test Cases:
{test_cases}

### Response:
"""


TASK_HEADER = "### Task"
GOAL_INSTRUCTION = (
    "### Your goal is to write a Python function that solves the problem above."
)
EXAMPLES_HEADER = "### Here are some examples:"

PROMPT_TEMPLATE = """{task_header}
{text}

{goal_instruction}
{examples_block}

{function_signature}
"""
