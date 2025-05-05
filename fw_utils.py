import requests
import pprint
import json
import sys
import re
from model_utils import convert_logprobs_dist_dict_to_tokenizer_prob_dist
from transformers import AutoTokenizer

FW_KEY = "fw_3ZkWSyAu3Z41GcwKkSM3pAax"
PROBLEM_PROMPT = """Write a function `count_vowels(s)` that takes a string `s` and returns the number of vowels (`a`, `e`, `i`, `o`, `u`) in the string. The function should be case-insensitive.

Examples:
count_vowels("hello") => 2
count_vowels("OpenAI") => 3
count_vowels("bcd") => 0

"""
LOGPROBS_COUNT = 5
HTTP_REQUEST_TO_LLM_RETRIES_COUNT = 3
REQUEST_TIMEOUT_SEC = 30
HF_DEEPSEEK_0324_MODEL_NAME = "deepseek-ai/DeepSeek-V3-0324"
FW_DEEPSEEK_0324_MODEL_NAME = "accounts/fireworks/models/deepseek-v3-0324"
FW_ENDPOINT_URL = "https://api.fireworks.ai/inference/v1/completions"


def extract_python_code(text):
    import re

    match = re.search(r"```python\n(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else None


def pseudo_beam_search_batch(
    prompt,
    s,
    model,
    url,
    headers,
    max_tokens=256,
    temperature=0.8,
    max_total_requests=10,
    batch_size=5,
):
    """
    Collects s unique code completions by calling the API in batches of n completions per request.

    Args:
        prompt (str): The generation prompt.
        s (int): Number of unique completions to gather.
        model (str): Model identifier.
        url (str): API endpoint.
        headers (dict): Authorization + content headers.
        max_tokens (int): Max tokens per completion.
        temperature (float): Sampling temperature.
        max_total_requests (int): Retry limit.
        batch_size (int): Number of completions per API call (n).

    Returns:
        List[str]: List of unique code completions (up to s).
    """
    seen = set()
    total_requests = 0

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {FW_KEY}",
    }
    print(f"[INFO] Starting pseudo beam search for {s} unique completions")

    while len(seen) < s and total_requests < max_total_requests:
        temperature = temperature * (1.02**total_requests)
        print(f"[INFO] Using batch size = {batch_size}, temperature = {temperature}")
        payload = {
            "model": model,
            "n": batch_size,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        try:
            print(f"[INFO] Sending request #{total_requests + 1}...")
            response = requests.post(
                url,
                headers=headers,
                data=json.dumps(payload),
                timeout=REQUEST_TIMEOUT_SEC,
            )
            if response.status_code != 200:
                raise RuntimeError(
                    f"Request failed: {response.status_code} - {response.text}"
                )

            data = response.json()
            for i, choice in enumerate(data["choices"]):
                raw_text = choice["text"]
                code = extract_python_code(raw_text)
                if code and code not in seen:
                    seen.add(code)
                    print(f"[SUCCESS] Added candidate #{len(seen)}")
                elif code:
                    print(f"[DUPLICATE] Skipping already seen completion #{i + 1}")
                else:
                    print(f"[WARN] No valid code block in completion #{i + 1}")

            total_requests += 1

        except Exception as e:
            print(f"[ERROR] Exception on request #{total_requests + 1}: {e}")
            total_requests += 1
            continue

    print(f"[INFO] Completed with {len(seen)} unique completions")
    return list(seen)


def get_next_token_top_logprob_dist(
    prompt, model, url, headers, logprobs_count=LOGPROBS_COUNT
):
    max_tokens = 1  ## should not be changed
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "logprobs": logprobs_count,
        "raw_output": True,
    }

    for _ in range(HTTP_REQUEST_TO_LLM_RETRIES_COUNT):
        try:
            response = requests.post(
                url,
                headers=headers,
                data=json.dumps(payload),
                timeout=REQUEST_TIMEOUT_SEC,
            )
            if response.status_code != 200:
                raise RuntimeError(
                    f"Request failed: {response.status_code} - {response.text}"
                )
            break
        except Exception as e:
            print(e)
            continue

    data = response.json()
    completion_tokens = data["usage"]["completion_tokens"]
    assert completion_tokens == max_tokens

    choice = data["choices"][0]
    raw_output_entry = choice["raw_output"]["completion_logprobs"]
    assert len(raw_output_entry["content"]) == max_tokens
    top_logprobs = raw_output_entry["content"][0]["top_logprobs"]
    assert len(top_logprobs) == logprobs_count
    top_logprobs = {entry["token_id"]: entry["logprob"] for entry in top_logprobs}
    return top_logprobs


def main():
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {FW_KEY}",
    }
    model = FW_DEEPSEEK_0324_MODEL_NAME
    url = FW_ENDPOINT_URL

    next_token_logprob_dist_dict = get_next_token_top_logprob_dist(
        prompt=PROBLEM_PROMPT,
        model=model,
        url=url,
        headers=headers,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        HF_DEEPSEEK_0324_MODEL_NAME, trust_remote_code=True
    )
    next_token_prob_dist = convert_logprobs_dist_dict_to_tokenizer_prob_dist(
        tokenizer, next_token_logprob_dist_dict
    )
    pprint.pprint(next_token_logprob_dist_dict)
    pprint.pprint(next_token_prob_dist)
    pprint.pprint(next_token_prob_dist.sum())
    return
    unique_codes = pseudo_beam_search_batch(
        prompt=PROBLEM_PROMPT,
        s=5,
        model=model,
        url=url,
        headers=headers,
        max_tokens=1024,
        temperature=0.9,
        max_total_requests=10,
        batch_size=5,
    )
    for i, code in enumerate(unique_codes):
        print(f"\n--- Candidate {i+1} ---\n{code}\n")


main()
