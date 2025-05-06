import requests
import pprint
import json
import re
import torch
from model_utils import convert_logprobs_dist_dict_to_tokenizer_prob_dist
from model_utils import extract_new_tokens, calculate_tokens_length
from code_generation_utils import CodeGenStopCriteria, prime_stopping_criteria
from transformers import AutoTokenizer
from transformers import StoppingCriteriaList

from consts import *

PROBLEM_PROMPT = """Write a function `count_vowels(s)` that takes a string `s` and returns the number of vowels (`a`, `e`, `i`, `o`, `u`) in the string. The function should be case-insensitive.

Examples:
count_vowels("hello") => 2
count_vowels("OpenAI") => 3
count_vowels("bcd") => 0

"""


class PostRequestTimeoutError(RuntimeError):
    pass


def extract_python_code(text):
    match = re.search(r"```python\n(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else None


def extract_prompt_python_code(text):
    python_code = extract_python_code(text)
    if python_code is not None:
        return python_code
    return text.split("```python\n")[1]


END_OF_CODE_STOP_SEQUENCE = "```\n"


def pseudo_beam_search_batch(
    prompt,
    tokenizer,
    execution_manager,
    unique_samples_count,
    nf_samples_depth,
    model_name,
    url,
    headers,
    max_tokens,
    temperature,
    max_total_requests,
    batch_size,
    crop_idx,
    top_p=0.95,
    post_requests_retries=HTTP_REQUEST_TO_LLM_RETRIES_COUNT,
):
    unique_codes = set()
    unique_executable_codes = set()
    total_requests = 0

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {FW_KEY}",
    }
    # print(
    #     f"[INFO] Starting pseudo beam search for {unique_samples_count} unique completions"
    # )
    # print(f"[INFO] Using batch size = {batch_size}, temperature = {temperature}")
    payload = {
        "model": model_name,
        "n": batch_size,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stop": END_OF_CODE_STOP_SEQUENCE,
    }
    total_completion_tokens = 0
    while (
        len(unique_codes) < unique_samples_count and total_requests < max_total_requests
    ):
        response = fw_utils__post_request_retries(
            url,
            headers,
            json.dumps(payload),
            timeout=REQUEST_TIMEOUT_SEC,
            post_requests_retries=post_requests_retries,
        )
        data = response.json()
        completion_tokens = data["usage"]["completion_tokens"]
        total_completion_tokens += completion_tokens
        prompt_input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
        prompt_new_text, _ = extract_new_tokens(tokenizer, prompt_input_ids, crop_idx)
        prompt_code = extract_prompt_python_code(prompt_new_text)
        prompt_code_lines_count = len(prompt_code.splitlines())
        for i, choice in enumerate(data["choices"]):
            raw_text = choice["text"]
            raw_text += END_OF_CODE_STOP_SEQUENCE
            full_answer = prompt + raw_text

            input_ids = tokenizer(full_answer, return_tensors="pt")["input_ids"]
            new_text, _ = extract_new_tokens(tokenizer, input_ids, crop_idx)
            full_code = extract_python_code(new_text)
            if not full_code:
                continue
            full_code = "\n".join(
                full_code.splitlines()[: (prompt_code_lines_count + nf_samples_depth)]
            )
            try:
                executable_partial_program_code = (
                    execution_manager.extract_partial_executable_program(full_code)
                )
            except ValueError:
                continue
            if (
                full_code
                and full_code not in unique_codes
                and executable_partial_program_code
                and executable_partial_program_code not in unique_executable_codes
            ):
                unique_codes.add(full_code)
                unique_executable_codes.add(executable_partial_program_code)
                # print(f"[SUCCESS] Added candidate #{len(unique_codes)}")
                # elif executable_partial_program_code:
                # print(f"[DUPLICATE] Skipping already seen completion #{i + 1}")
                # else:
                #     pass
                # print(f"[WARN] No valid code block in completion #{i + 1}")
            if len(unique_codes) >= unique_samples_count:
                break
        total_requests += 1
        temperature = temperature * (1.02**total_requests)

    # print(f"[INFO] Completed with {len(unique_codes)} unique completions")
    unique_codes = list(unique_codes)
    return unique_codes, total_completion_tokens




def fw_utils__post_request_retries(url, headers, data, timeout, post_requests_retries):
    for retry_idx in range(post_requests_retries):
        err = None
        response = None
        try:
            # print(f"[INFO] Sending request #{retry_idx + 1}")
            response = requests.post(
                url,
                headers=headers,
                data=data,
                timeout=REQUEST_TIMEOUT_SEC,
            )
            if response.status_code != HTTP_SUCCESS_CODE:
                print(
                    f"[ERROR] Exception on request #{retry_idx + 1}:\nCode: {response.status_code}: {response.text}"
                )
                continue
            # Success
            break
        except Exception as e:
            err = e
            print(f"[ERROR] Exception on request #{retry_idx + 1}: {e}")
            continue
    if err or (response and response.status_code != HTTP_SUCCESS_CODE):
        raise PostRequestTimeoutError(
            f"Request failed after {post_requests_retries} times"
        )
    return response


def fw_utils__get_next_token_top_logprob_dist(
    prompt,
    model_name,
    url,
    headers,
    logprobs_count=LOGPROBS_COUNT,
    post_requests_retries=HTTP_REQUEST_TO_LLM_RETRIES_COUNT,
):
    max_tokens = 1  ## should not be changed
    mapped_model_name = HF_MODEL_TO_FW_MODEL[model_name]
    payload = {
        "model": mapped_model_name,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "logprobs": logprobs_count,
        "raw_output": True,
    }
    response = fw_utils__post_request_retries(
        url,
        headers,
        json.dumps(payload),
        timeout=REQUEST_TIMEOUT_SEC,
        post_requests_retries=post_requests_retries,
    )
    data = response.json()
    completion_tokens = data["usage"]["completion_tokens"]
    assert completion_tokens == max_tokens

    choice = data["choices"][0]
    raw_output_entry = choice["raw_output"]["completion_logprobs"]
    assert len(raw_output_entry["content"]) == max_tokens
    top_logprobs = raw_output_entry["content"][0]["top_logprobs"]
    assert len(top_logprobs) == logprobs_count
    top_logprobs = {entry["token_id"]: entry["logprob"] for entry in top_logprobs}
    return top_logprobs, completion_tokens


def fw_utils__sample_code_pseudo_beam_search(
    input_ids,
    tokenizer,
    execution_manager,
    stats_manager,
    samples_count,
    temperature,
    nf_samples_depth,
    crop_idx,
    model_name=DEEPSEEK_0324_MODEL_NAME_FW,
    url=FW_ENDPOINT_URL,
    fw_key=FW_KEY,
    max_tokens=PSEUDO_BEAM_SEARCH_MAX_TOKENS,
    max_total_requests=PSEUDO_BEAM_SEARCH_MAX_TOTAL_REQUESTS,
):
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {fw_key}",
    }
    prompt = tokenizer.batch_decode(input_ids, skip_special_tokens=True)[0]
    unique_codes, total_completion_tokens = pseudo_beam_search_batch(
        prompt,
        tokenizer,
        execution_manager,
        samples_count,
        nf_samples_depth,
        model_name,
        url,
        headers,
        max_tokens,
        temperature,
        max_total_requests,
        samples_count,
        crop_idx,
    )
    if stats_manager is not None:
        stats_manager.increate_counter("beam_search_input_tokens", input_ids.shape[1])
        stats_manager.increate_counter(
            "beam_search_output_tokens", total_completion_tokens
        )
    return unique_codes


CODE_BORDER_TOKEN = "```"
END_OF_SENTENCE_TOKEN = "<__end_of_sentence__>"


def inference_endpoint_dsgi(
    prompt,
    tokenizer,
    model_name,
    dsgi_injection_manager,
    max_tokens=PSEUDO_BEAM_SEARCH_MAX_TOKENS,
    debug=True,
    do_sample=False,
):
    stats_manager = dsgi_injection_manager.adapter.stats_manager
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    new_text = ""
    code_borders_tokens_count = 0
    is_dsgi_enabled = False
    end_of_sentence_token_id = tokenizer.encode(
        END_OF_SENTENCE_TOKEN, add_special_tokens=False
    )[0]
    previous_executable_partial_program_code = None
    executable_partial_program_code, new_code = None, None
    for _ in range(max_tokens):
        # current_prompt = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        # print(current_prompt)
        # print()
        original_probs = fw_utils__get_next_token_prob_dist(
            input_ids, tokenizer, model_name, stats_manager=stats_manager
        )
        probs = original_probs

        #### Extract Dynamic Signal  ####
        is_dsgi_enabled = (dsgi_injection_manager is not None) and (
            dsgi_injection_manager.is_dsgi_enabled(input_ids.clone())
        )
        if is_dsgi_enabled:
            dynamic_signal_input_ids, debug_data = (
                dsgi_injection_manager.extract_dynamic_signal_input_ids(
                    input_ids.clone()
                )
            )
            # no dynamic signals were extracted, no need for guidance
            if torch.equal(dynamic_signal_input_ids, input_ids):
                is_dsgi_enabled = False
            if debug:
                executable_partial_program_code, new_code = debug_data
                if (
                    previous_executable_partial_program_code
                    != executable_partial_program_code
                ):
                    # print("#" * 10)
                    # print(new_code)
                    # print()
                    # print("$" * 10)
                    # print(new_text)
                    # print("#" * 10)
                    # print()
                    previous_executable_partial_program_code = (
                        executable_partial_program_code
                    )
        ###########
        if is_dsgi_enabled:
            #### Calculate Dynamic Signal conditional distibution ####
            dyn_probs = fw_utils__get_next_token_prob_dist(
                dynamic_signal_input_ids,
                tokenizer,
                model_name,
                stats_manager=stats_manager,
            )

            #### Apply Dynamic Signal Guidance ####
            probs_guided = dsgi_injection_manager.apply_guidance(
                original_probs, dyn_probs, debug=False
            )
            probs = probs_guided
        #########################

        # Next Token Selection
        if do_sample:
            next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_token = torch.argmax(probs, dim=-1)

        next_token_text = tokenizer.decode(next_token)
        new_text += next_token_text
        # print(next_token, next_token_text, code_borders_tokens_count)
        if CODE_BORDER_TOKEN in next_token_text:
            code_borders_tokens_count += 1
        if code_borders_tokens_count >= 2:
            break
        if next_token.item() == end_of_sentence_token_id:
            break
        input_ids = torch.cat([input_ids, next_token[:, None]], dim=-1)

    input_ids = input_ids.squeeze(0)
    outputs = [input_ids]
    return outputs


def fw_utils__get_next_token_prob_dist(
    input_ids,
    tokenizer,
    model_name,
    fw_key=FW_KEY,
    url=FW_ENDPOINT_URL,
    stats_manager=None,
):
    prompt = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {fw_key}",
    }
    next_token_logprob_dist_dict, completion_tokens = (
        fw_utils__get_next_token_top_logprob_dist(
            prompt, model_name, url=url, headers=headers
        )
    )
    next_token_prob_dist = convert_logprobs_dist_dict_to_tokenizer_prob_dist(
        tokenizer, next_token_logprob_dist_dict
    )
    next_token_prob_dist = next_token_prob_dist.unsqueeze(0)

    if stats_manager is not None:
        stats_manager.increate_counter("guidance_input_tokens", input_ids.shape[1])
        stats_manager.increate_counter("guidance_output_tokens", completion_tokens)
    return next_token_prob_dist
