import os
import traceback
import requests
import re
from collections import defaultdict
import json
import torch
from model_utils import convert_logprobs_dist_dict_to_tokenizer_prob_dist

from consts import *


class PostRequestTimeoutError(RuntimeError):
    pass


def extract_python_code(text):
    match = re.search(r"```python\n(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else None


def extract_prompt_python_code(text):
    python_code = extract_python_code(text)
    if python_code is not None:
        return python_code
    return text.split(f"{INSTRUCT_MODEL_PYTHON_CODE_START}")[1]


def extract_function_name(signature_line):
    match = re.match(r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", signature_line)
    return match.group(1) if match else None


def extract_matching_blocks(text, verbose=False):
    pattern = r"```python\n(.*?)```"
    matches = list(re.finditer(pattern, text, re.DOTALL))
    if verbose:
        print(f"Found {len(matches)} code blocks.\n")
    return matches


def extract_matching_blocks_with_def_index(matches, target_func_name, verbose=False):
    def_index_dict = defaultdict(list)  # block_str -> list of def_start_indices
    last_block = None

    for i, match in enumerate(matches):
        block = match.group(1)
        block_start_in_text = match.start(1)
        lines = block.splitlines()

        idx_sum = 0
        for rel_line_idx, line in enumerate(lines):
            print(f"{i}:{rel_line_idx}: {line}")
            line_stripped = line.strip()
            if (
                not line_stripped
                or line_stripped.startswith("#")
                or line_stripped.startswith("import")
                or line_stripped.startswith("from")
            ):
                idx_sum += len(line + "\n")
                continue

            if re.match(rf"^def\s+{target_func_name}\s*\(", line_stripped):
                def_index = block_start_in_text + idx_sum
                block_cleaned = block.strip()
                def_index_dict[block_cleaned].append(def_index)
                last_block = block_cleaned

                print(
                    f"Block {i} matched\n{block}\n`{target_func_name}` at index {def_index}."
                )
                break
            else:
                print(f"Block {i} not matched:\n{block}\n")
                break

    final_def_index = def_index_dict[last_block][-1] if last_block else None
    if verbose:
        print(
            f"\nFirst occurrence of last matching block starts at index {final_def_index}."
        )
        print(f"Total unique blocks: {len(def_index_dict)}")

    return final_def_index, last_block


def reasoning_tokens_query(
    prompt,
    function_signature,
    model_name,
    temperture,
    max_tokens,
    top_p=FW_UTILS__DEFAULT_TOP_P,
    post_requests_retries=HTTP_REQUEST_TO_LLM_RETRIES_COUNT,
    verbose=False,
    stop_condition=COMPLEX_QUERY_STOP_CONDITION,
    function_name=None,
    return_raw=False,
):
    inference_endpoint_api_key = os.environ.get("FW_KEY")
    inference_endpoint_url = os.environ.get("FW_ENDPOINT_URL")
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {inference_endpoint_api_key}",
    }
    total_match_retries = MATCH_RETRIES_COUNT
    total_completion_tokens = 0
    answer_start_until_code = None
    for match_retry in range(total_match_retries):
        temperture = temperture * (0.9**match_retry)
        payload = {
            "model": HF_MODEL_TO_FW_MODEL[model_name],
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperture,
            "top_p": top_p,
            "stop": stop_condition,
        }
        try:
            print(f"Match Retry #{match_retry + 1}/{total_match_retries}")
            response = inference_endpoint_utils__post_request_retries(
                inference_endpoint_url,
                headers,
                json.dumps(payload),
                timeout=QWEN_REQUEST_TIMEOUT_SEC,
                post_requests_retries=post_requests_retries,
                verbose=verbose,
            )
            if type(response) == type(""):
                raw_text = response
                completion_tokens = 0
            else:
                data = response.json()
                raw_text = data["choices"][0]["text"]
                assert data.get("usage", None), f"Invalid response: {data}"
                completion_tokens = data["usage"]["completion_tokens"]
        except Exception as e:
            general_error = str(type(e))
            tb = traceback.format_exc()
            print(f"Error: {general_error}")
            print(tb)
            print()
            continue

        if return_raw:
            answer_start_until_code = ""
            last_block = raw_text
            return answer_start_until_code, last_block, completion_tokens

        total_completion_tokens += completion_tokens
        raw_text = raw_text.replace(
            "<python>", INSTRUCT_MODEL_PYTHON_CODE_START_TOK
        ).replace("</python>", CODE_BORDER_TOKEN)
        if function_name is None:
            function_name = extract_function_name(function_signature)
        matches = extract_matching_blocks(raw_text, verbose=True)
        if not matches:
            continue

        answer_start_idx, last_block = extract_matching_blocks_with_def_index(
            matches, function_name, verbose=True
        )
        if not answer_start_idx:
            continue
        answer_start_until_code = (
            raw_text[:answer_start_idx] if answer_start_idx is not None else None
        )
        if not answer_start_until_code:
            continue
        # print("\n=== Answer Until Code ===")
        # print(answer_start_until_code)
        break

    assert answer_start_until_code
    return answer_start_until_code, last_block, total_completion_tokens


def simple_query(
    prompt,
    model_name,
    temperture,
    top_p=FW_UTILS__DEFAULT_TOP_P,
    max_tokens=PSEUDO_BEAM_SEARCH_MAX_TOKENS,
    post_requests_retries=HTTP_REQUEST_TO_LLM_RETRIES_COUNT,
    stop_condition=END_OF_CODE_STOP_SEQUENCE,
    extract_code=True,
    add_stop_condition=True,
    verbose=False,
):
    inference_endpoint_api_key = os.environ.get("FW_KEY")
    inference_endpoint_url = os.environ.get("FW_ENDPOINT_URL")
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {inference_endpoint_api_key}",
    }
    payload = {
        "model": HF_MODEL_TO_FW_MODEL[model_name],
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperture,
        "top_p": top_p,
        "stop": stop_condition,
    }
    response = inference_endpoint_utils__post_request_retries(
        inference_endpoint_url,
        headers,
        json.dumps(payload),
        timeout=REQUEST_TIMEOUT_SEC,
        post_requests_retries=post_requests_retries,
        verbose=verbose,
    )
    data = response.json()
    assert data.get("usage", None), f"Invalid response: {data}"
    completion_tokens = data["usage"]["completion_tokens"]
    raw_text = data["choices"][0]["text"]
    if add_stop_condition:
        raw_text += stop_condition
    output = raw_text
    if extract_code:
        output = extract_python_code(raw_text)
    return output, completion_tokens


def beam_search_batch(
    prompt,
    execution_manager,
    unique_candidates_count,
    bs_completion_horizon,
    model_name,
    max_tokens,
    temperature,
    batch_size,
    prompt_with_cot,
    max_total_requests=PSEUDO_BEAM_SEARCH_MAX_TOTAL_REQUESTS,
    top_p=FW_UTILS__DEFAULT_TOP_P,
    post_requests_retries=HTTP_REQUEST_TO_LLM_RETRIES_COUNT,
):
    inference_endpoint_api_key = os.environ.get("FW_KEY")
    inference_endpoint_url = os.environ.get("FW_ENDPOINT_URL")

    unique_codes = set()
    unique_executable_codes = set()
    total_requests = 0

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {inference_endpoint_api_key}",
    }
    print(
        f"[INFO] Starting beam search for {unique_candidates_count} unique completions"
    )
    total_completion_tokens = 0
    while (
        len(unique_codes) < unique_candidates_count
        and total_requests < max_total_requests
    ):
        n = batch_size * (total_requests + 1)
        payload = {
            "model": HF_MODEL_TO_FW_MODEL[model_name],
            "n": n,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stop": END_OF_CODE_STOP_SEQUENCE,
        }
        response = inference_endpoint_utils__post_request_retries(
            inference_endpoint_url,
            headers,
            json.dumps(payload),
            timeout=REQUEST_TIMEOUT_SEC,
            post_requests_retries=post_requests_retries,
            verbose=True,
        )
        data = response.json()
        assert data.get("usage", None), f"Invalid response: {data}"
        completion_tokens = data["usage"]["completion_tokens"]
        total_completion_tokens += completion_tokens
        only_answer = prompt[len(prompt_with_cot) :]
        if INSTRUCT_MODEL_PYTHON_CODE_START in only_answer:
            only_answer = only_answer.split(INSTRUCT_MODEL_PYTHON_CODE_START)[1]
        choices = [choice for choice in data["choices"]]
        unique_choices = []
        for choice in choices:
            if choice not in unique_choices:
                unique_choices.append(choice)
        print(
            f"[INFO] Using batch size = {batch_size}, d={bs_completion_horizon} t={temperature}"
        )
        print(
            f"[INFO] unique_samples={len(unique_choices)}/{len(choices)}, current_unique_count={len(unique_codes)}/{unique_candidates_count}"
        )
        for i, choice in enumerate(unique_choices):
            raw_text = choice["text"]

            # now we crop until ```
            raw_text = raw_text.split(CODE_BORDER_TOKEN)[0]

            # now we split and take only the depth
            new_code_lines = raw_text.splitlines()
            full_code = only_answer
            for idx, line in enumerate(new_code_lines):
                if idx >= bs_completion_horizon + 1:
                    break
                if CODE_BORDER_TOKEN in line:
                    break
                if END_OF_SENTENCE_TOKEN in line:
                    break
                if END_OF_TEXT_TOKEN in line:
                    break
                full_code += line + "\n"
            full_code = full_code.replace(CODE_BORDER_TOKEN, "")
            full_code = full_code.replace(END_OF_SENTENCE_TOKEN, "")
            full_code = full_code.replace(END_OF_TEXT_TOKEN, "")

            full_code_lines = full_code.splitlines()
            for line in full_code_lines:
                if not line:
                    continue
                start_condition = line.strip().startswith(
                    ("def", "import", "from", "#")
                )
                break

            if not start_condition:
                print("#" * 10 + "    Invalid Code    " + "#" * 10)
                print("=" * 10)
                print(raw_text)
                print("=" * 10)
                continue

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
                print(f"#{i}")
                print("#" * 10 + "    Full Code    " + "#" * 10)
                print(f"Length: {len(full_code.splitlines())} lines")
                print(full_code)
                print("#" * 20)
                print("#" * 10 + " Executable Code " + "#" * 10)
                print(
                    f"Length: {len(executable_partial_program_code.splitlines())} lines"
                )
                print(executable_partial_program_code)
                print("#" * 20)
                print()
            if len(unique_codes) >= unique_candidates_count:
                break
        total_requests += 1
        temperature = temperature * (1.2**total_requests)

    print(
        f"[INFO] Completed with {len(unique_codes)}/{unique_candidates_count} unique completions"
    )
    unique_codes = list(unique_codes)
    return unique_codes, total_completion_tokens


def inference_endpoint_utils__post_request_retries(
    url,
    headers,
    data,
    post_requests_retries,
    verbose=False,
    timeout=REQUEST_TIMEOUT_SEC,
):
    initial_timeout = timeout
    for retry_idx in range(post_requests_retries):
        err = None
        response = None
        timeout = initial_timeout * (2 * retry_idx + 1)
        try:
            if verbose:
                print(
                    f"[INFO] Sending request #{retry_idx + 1}/{post_requests_retries} (timeout={timeout}sec)"
                )
            FW_API = True
            TOGETHER_API = False
            if FW_API:
                response = requests.post(
                    url,
                    headers=headers,
                    data=data,
                    timeout=timeout,
                )
                continue

            respones_data = response.json()
            assert respones_data["usage"], f"Invalid response: {respones_data}"
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


def inference_endpoint_utils__get_next_token_top_logprob_dist(
    prompt,
    model_name,
    url,
    headers,
    logprobs_count=LOGPROBS_COUNT,
    post_requests_retries=HTTP_REQUEST_TO_LLM_RETRIES_COUNT,
):
    MAX_TOKENS__DONT_CHANGE = 1
    max_tokens = MAX_TOKENS__DONT_CHANGE  ## should not be changed
    assert max_tokens == 1
    payload = {
        "model": HF_MODEL_TO_FW_MODEL[model_name],
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "logprobs": logprobs_count,
        "raw_output": True,
    }
    response = inference_endpoint_utils__post_request_retries(
        url,
        headers,
        json.dumps(payload),
        timeout=REQUEST_TIMEOUT_SEC,
        post_requests_retries=post_requests_retries,
    )
    data = response.json()
    try:
        assert data.get("usage", None), f"Invalid response: {data}"
        completion_tokens = data["usage"]["completion_tokens"]
        assert completion_tokens == max_tokens
    except:
        completion_tokens = 0

    choice = data["choices"][0]
    raw_output_entry = choice["raw_output"]["completion_logprobs"]
    assert len(raw_output_entry["content"]) == max_tokens
    top_logprobs = raw_output_entry["content"][0]["top_logprobs"]
    assert len(top_logprobs) == logprobs_count
    top_logprobs = {entry["token_id"]: entry["logprob"] for entry in top_logprobs}
    return top_logprobs, completion_tokens


def inference_endpoint_utils__sample_code_beam_search(
    input_ids,
    tokenizer,
    execution_manager,
    stats_manager,
    candidates_count,
    temperature,
    bs_completion_horizon,
    prompt_with_cot,
    model_name,
    max_tokens=PSEUDO_BEAM_SEARCH_MAX_TOKENS,
):
    batch_size = max(FW__MIN_BATCH_SIZE, candidates_count)
    prompt = tokenizer.batch_decode(input_ids, skip_special_tokens=True)[0]
    unique_codes, total_completion_tokens = beam_search_batch(
        prompt=prompt,
        execution_manager=execution_manager,
        unique_candidates_count=candidates_count,
        bs_completion_horizon=bs_completion_horizon,
        model_name=model_name,
        max_tokens=max_tokens,
        temperature=temperature,
        max_total_requests=PSEUDO_BEAM_SEARCH_MAX_TOTAL_REQUESTS,
        batch_size=batch_size,
        prompt_with_cot=prompt_with_cot,
    )
    if stats_manager is not None:
        stats_manager.increate_counter("beam_search_input_tokens", input_ids.shape[1])
        stats_manager.increate_counter(
            "beam_search_output_tokens", total_completion_tokens
        )
    return unique_codes


def extract_eg_cfg_start_prefix(
    prompt, model_name, eg_cfg_injection_manager, function_signature, function_name=None
):
    assert model_name in (DEEPSEEK_V3_0324_MODEL_NAME_HF, QWEN3_253B_MODEL_NAME_HF)
    answer_start_until_code, _, completion_tokens = reasoning_tokens_query(
        prompt,
        function_signature,
        model_name,
        temperture=eg_cfg_injection_manager.adapter.temperature,
        max_tokens=REASONING_TOKENS_QUERY_MAX_TOKENS,
        verbose=True,
        function_name=function_name,
    )
    return answer_start_until_code, completion_tokens


def inference_endpoint_eg_cfg_gamma_1_optimization(
    prompt,
    tokenizer,
    model_name,
    eg_cfg_injection_manager,
    function_signature,
    max_tokens=PSEUDO_BEAM_SEARCH_MAX_TOKENS,
    debug=True,
    function_name=None,
):
    inference_initial_prompt_input_ids_len = None
    stats_manager = eg_cfg_injection_manager.adapter.stats_manager
    new_text = ""
    code_borders_tokens_count = 0
    is_eg_cfg_enabled = False
    end_of_sentence_token_id = tokenizer.encode(
        END_OF_SENTENCE_TOKEN, add_special_tokens=False
    )[0]
    previous_executable_partial_program_code = None
    executable_partial_program_code, new_code = None, None

    answer_start_until_code, completion_tokens = extract_eg_cfg_start_prefix(
        prompt, model_name, eg_cfg_injection_manager, function_signature, function_name
    )

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    if stats_manager is not None:
        stats_manager.increate_counter("guidance_input_tokens", input_ids.shape[1])
        stats_manager.increate_counter("guidance_output_tokens", completion_tokens)

    prompt += answer_start_until_code
    eg_cfg_injection_manager.adapter.prompt_with_cot = prompt

    inference_initial_prompt_input_ids_len = tokenizer(prompt, return_tensors="pt")[
        "input_ids"
    ].shape[1]
    if model_name in (QWEN3_253B_MODEL_NAME_HF, DEEPSEEK_V3_0324_MODEL_NAME_HF):
        # now we have the starting ```python
        if function_name is None:
            function_name = extract_function_name(function_signature)
        prompt += f"def {function_name}("
        new_text += f"def {function_name}("
        code_borders_tokens_count += 1
        if function_name == "solve":
            prompt += f"):"
            new_text += f"):"

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    early_stop = False
    initial_tokens_conunt = input_ids.shape[1]
    while input_ids.shape[1] < initial_tokens_conunt + max_tokens:
        #### Extract Dynamic Signal  ####
        is_eg_cfg_enabled = (eg_cfg_injection_manager is not None) and (
            eg_cfg_injection_manager.is_eg_cfg_enabled(input_ids.clone())
        )
        if is_eg_cfg_enabled:
            dynamic_signal_input_ids, debug_data = (
                eg_cfg_injection_manager.extract_dynamic_signal_input_ids(
                    input_ids.clone()
                )
            )
            # no dynamic signals were extracted, no need for guidance
            if torch.equal(dynamic_signal_input_ids, input_ids):
                is_eg_cfg_enabled = False
            if debug:
                executable_partial_program_code, new_code = debug_data
                if (
                    previous_executable_partial_program_code
                    != executable_partial_program_code
                ):
                    # promp_without_signal = tokenizer.decode(input_ids[0])
                    # promp_with_signal = tokenizer.decode(dynamic_signal_input_ids[0])
                    previous_executable_partial_program_code = (
                        executable_partial_program_code
                    )
                    lines_count = len(new_text.splitlines())
                    print(f"Current Code: {lines_count} lines")
                    print(new_text)
                    print()
        ###########
        ## get next line
        next_line = ""
        if new_text.count("\n") > 0:
            output, _ = inference_endpoint_utils__get_next_line(
                dynamic_signal_input_ids,
                tokenizer,
                model_name,
                stats_manager=stats_manager,
            )
            lines = output.splitlines()  # take only the first line
            next_line = lines[0]
        next_line = next_line + "\n"
        next_line_tokens = tokenizer.encode(next_line)

        # convert to tokens
        do_break = False
        for next_token in next_line_tokens:
            #########################
            if next_token == 0:
                continue
            next_token_text = tokenizer.decode(next_token)
            new_text += next_token_text
            # print(next_token_text)
            # print(next_token, [next_token_text], code_borders_tokens_count)

            if CODE_BORDER_TOKEN in next_token_text:
                code_borders_tokens_count += 1
            if code_borders_tokens_count >= 2:
                do_break = True
                break
            if next_token == end_of_sentence_token_id:
                do_break = True
                break
            if END_OF_SENTENCE_TOKEN in next_token_text:
                do_break = True
                break
            next_token = torch.tensor([next_token], device=input_ids.device)
            input_ids = torch.cat([input_ids, next_token[:, None]], dim=-1)
            if eg_cfg_injection_manager.early_stop_detected():
                if eg_cfg_injection_manager.gamma != 1.0:
                    solution_code = (
                        eg_cfg_injection_manager.adapter.early_stop_detected_program
                    )
                else:
                    solution_code = (
                        eg_cfg_injection_manager.adapter.dynamic_early_stop_detected_program
                    )
                early_stop = True
                return solution_code, early_stop
            print(new_text)
        if do_break:
            break

    inference_initial_prompt_input_ids_len = None
    return input_ids, early_stop, inference_initial_prompt_input_ids_len


def inference_endpoint_eg_cfg(
    prompt,
    tokenizer,
    model_name,
    eg_cfg_injection_manager,
    function_signature,
    max_tokens=PSEUDO_BEAM_SEARCH_MAX_TOKENS,
    debug=True,
    do_sample=False,
    function_name=None,
):
    inference_initial_prompt_input_ids_len = None
    stats_manager = eg_cfg_injection_manager.adapter.stats_manager
    new_text = ""
    code_borders_tokens_count = 0
    is_eg_cfg_enabled = False
    end_of_sentence_token_id = tokenizer.encode(
        END_OF_SENTENCE_TOKEN, add_special_tokens=False
    )[0]
    previous_executable_partial_program_code = None
    executable_partial_program_code, new_code = None, None

    answer_start_until_code, completion_tokens = extract_eg_cfg_start_prefix(
        prompt, model_name, eg_cfg_injection_manager, function_signature, function_name
    )

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    if stats_manager is not None:
        stats_manager.increate_counter("guidance_input_tokens", input_ids.shape[1])
        stats_manager.increate_counter("guidance_output_tokens", completion_tokens)
    prompt += answer_start_until_code
    eg_cfg_injection_manager.adapter.prompt_with_cot = prompt

    inference_initial_prompt_input_ids_len = tokenizer(prompt, return_tensors="pt")[
        "input_ids"
    ].shape[1]
    if model_name in (QWEN3_253B_MODEL_NAME_HF, DEEPSEEK_V3_0324_MODEL_NAME_HF):
        # now we have the starting ```python
        if function_name is None:
            function_name = extract_function_name(function_signature)
        prompt += f"def {function_name}("
        new_text += f"def {function_name}("
        code_borders_tokens_count += 1
        if function_name == "solve":
            prompt += f"):"
            new_text += f"):"

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    early_stop = False
    for _ in range(max_tokens):
        #### Extract Dynamic Signal  ####
        is_eg_cfg_enabled = (eg_cfg_injection_manager is not None) and (
            eg_cfg_injection_manager.is_eg_cfg_enabled(input_ids.clone())
        )
        if is_eg_cfg_enabled:
            dynamic_signal_input_ids, debug_data = (
                eg_cfg_injection_manager.extract_dynamic_signal_input_ids(
                    input_ids.clone()
                )
            )
            # no dynamic signals were extracted, no need for guidance
            if torch.equal(dynamic_signal_input_ids, input_ids):
                is_eg_cfg_enabled = False
            if debug:
                executable_partial_program_code, new_code = debug_data
                if (
                    previous_executable_partial_program_code
                    != executable_partial_program_code
                ):
                    # promp_without_signal = tokenizer.decode(input_ids[0])
                    # promp_with_signal = tokenizer.decode(dynamic_signal_input_ids[0])
                    previous_executable_partial_program_code = (
                        executable_partial_program_code
                    )
                    lines_count = len(new_text.splitlines())
                    print(f"Current Code: {lines_count} lines")
                    print(new_text)
                    print()
        ###########
        if not (is_eg_cfg_enabled and eg_cfg_injection_manager.gamma == 1.0):
            original_probs = inference_endpoint_utils__get_next_token_prob_dist(
                input_ids, tokenizer, model_name, stats_manager=stats_manager
            )
            probs = original_probs
        else:
            # print("Optimization for gamma=1 is enabled")
            original_probs = None

        if is_eg_cfg_enabled:
            #### Calculate Dynamic Signal conditional distibution ####
            dyn_probs = inference_endpoint_utils__get_next_token_prob_dist(
                dynamic_signal_input_ids,
                tokenizer,
                model_name,
                stats_manager=stats_manager,
            )

            if eg_cfg_injection_manager.gamma != 1.0:
                #### Apply Dynamic Signal Guidance ####
                probs_guided = eg_cfg_injection_manager.apply_guidance(
                    original_probs, dyn_probs, debug=False
                )
            else:
                # print("Use dynamic probs, no CFG needed")
                probs_guided = dyn_probs
            probs = probs_guided
        #########################

        # Next Token Selection
        if do_sample:
            next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_token = torch.argmax(probs, dim=-1)

        next_token_text = tokenizer.decode(next_token)
        new_text += next_token_text
        # print(next_token_text)
        # print(next_token, [next_token_text], code_borders_tokens_count)

        if CODE_BORDER_TOKEN in next_token_text:
            code_borders_tokens_count += 1
        if code_borders_tokens_count >= 2:
            break
        if next_token.item() == end_of_sentence_token_id:
            break
        if "<｜end▁of▁sentence｜>" in next_token_text:
            break
        if END_OF_SENTENCE_TOKEN in next_token_text:
            break

        input_ids = torch.cat([input_ids, next_token[:, None]], dim=-1)
        if eg_cfg_injection_manager.early_stop_detected():
            if eg_cfg_injection_manager.gamma != 1.0:
                solution_code = (
                    eg_cfg_injection_manager.adapter.early_stop_detected_program
                )
            else:
                solution_code = (
                    eg_cfg_injection_manager.adapter.dynamic_early_stop_detected_program
                )
            early_stop = True
            return solution_code, early_stop

    inference_initial_prompt_input_ids_len = None
    return input_ids, early_stop, inference_initial_prompt_input_ids_len


def inference_endpoint_utils__get_next_line(
    input_ids, tokenizer, model_name, stats_manager=None, max_tokens=64
):
    inference_endpoint_api_key = os.environ.get("FW_KEY")
    inference_endpoint_url = os.environ.get("FW_ENDPOINT_URL")
    prompt = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {inference_endpoint_api_key}",
    }
    payload = {
        "model": HF_MODEL_TO_FW_MODEL[model_name],
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "top_p": 0.95,
        "stop": COMPLEX_QUERY_STOP_CONDITION,
    }
    response = inference_endpoint_utils__post_request_retries(
        inference_endpoint_url,
        headers,
        json.dumps(payload),
        timeout=REQUEST_TIMEOUT_SEC,
        post_requests_retries=HTTP_REQUEST_TO_LLM_RETRIES_COUNT,
        verbose=False,
    )
    data = response.json()
    assert data.get("usage", None), f"Invalid response: {data}"
    completion_tokens = data["usage"]["completion_tokens"]
    raw_text = data["choices"][0]["text"]
    output = raw_text
    return output, completion_tokens


def inference_endpoint_utils__get_next_token_prob_dist(
    input_ids,
    tokenizer,
    model_name,
    stats_manager=None,
):
    inference_endpoint_api_key = os.environ.get("FW_KEY")
    inference_endpoint_url = os.environ.get("FW_ENDPOINT_URL")
    prompt = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {inference_endpoint_api_key}",
    }
    next_token_logprob_dist_dict, completion_tokens = (
        inference_endpoint_utils__get_next_token_top_logprob_dist(
            prompt, model_name, url=inference_endpoint_url, headers=headers
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
