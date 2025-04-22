import sys
import re
import os
from argparse import Namespace

# --- Constants ---
PROMPT_TYPE__DEEPSEEK_BASE = "deepseek_base"
PROMPT_TYPE__DEEPSEEK_INSTRUCT = "deepseek_instruct"
PROMPT_TYPE__INSTRUCT_LONG_CODE_PROMPT = "long_code"

GUIDANCE_STRATEGY__TOKEN_GUIDANCE = "token_guidance"
GUIDANCE_STRATEGY__PERSISTENT_PREFIX_GUIDANCE = "persistent_prefix_guidance"
GUIDANCE_STRATEGY__LINE_GUIDANCE = "line_guidance"

DYNAMIC_SIGNAL__PARTIAL_EXECUTION = "PartialExecution"
DYNAMIC_SIGNAL__NEAREST_FUTURE_EXECUTION = "NearestFutureExecution"
DYNAMIC_SIGNAL__BACKWARD = "Backward"


# --- Original Builder ---
def get_dynamic_signals(args):
    dynamic_signals_str = []
    dynamic_signals = []
    prompt_type_prefix = None
    guidance_strategy_prefix = None

    if args.prompt_type == PROMPT_TYPE__DEEPSEEK_INSTRUCT:
        prompt_type_prefix = "dsi"
    if args.prompt_type == PROMPT_TYPE__DEEPSEEK_BASE:
        prompt_type_prefix = "dsb"
    if args.prompt_type == PROMPT_TYPE__INSTRUCT_LONG_CODE_PROMPT:
        prompt_type_prefix = "lci"
    assert prompt_type_prefix, "Invalid Prompt Type"

    if args.g == GUIDANCE_STRATEGY__TOKEN_GUIDANCE:
        guidance_strategy_prefix = "tok"
    if args.g == GUIDANCE_STRATEGY__LINE_GUIDANCE:
        guidance_strategy_prefix = "ln"
    if args.g == GUIDANCE_STRATEGY__PERSISTENT_PREFIX_GUIDANCE:
        guidance_strategy_prefix = "prf"

    if args.p:
        dynamic_signals_str.append("p")
        dynamic_signals.append(DYNAMIC_SIGNAL__PARTIAL_EXECUTION)
    if args.n:
        d_arg = "inf"
        if args.d is not None:
            d_arg = str(args.d)
        dynamic_signals_str.append(f"ns{args.s}t{args.t}d{d_arg}")
        dynamic_signals.append(DYNAMIC_SIGNAL__NEAREST_FUTURE_EXECUTION)
    if args.b:
        dynamic_signals_str.append("b")
        dynamic_signals.append(DYNAMIC_SIGNAL__BACKWARD)

    dynamic_signals_str = "".join(dynamic_signals_str)
    if args.prompt_type == PROMPT_TYPE__INSTRUCT_LONG_CODE_PROMPT:
        dynamic_signals_str = f"{dynamic_signals_str}_{prompt_type_prefix}"
    dynamic_signals_str = f"{dynamic_signals_str}_{guidance_strategy_prefix}"
    dynamic_signals = tuple(dynamic_signals)
    assert dynamic_signals
    return dynamic_signals, dynamic_signals_str


# --- Reverse Parser ---
def parse_dynamic_signals_str(s):
    result = {
        "prompt_type": None,
        "g": None,
        "p": False,
        "n": False,
        "d": None,
        "s": None,
        "t": None,
        "b": False,
    }

    if s.endswith("_tok"):
        result["g"] = GUIDANCE_STRATEGY__TOKEN_GUIDANCE
        suffix = "_tok"
    elif s.endswith("_ln"):
        result["g"] = GUIDANCE_STRATEGY__LINE_GUIDANCE
        suffix = "_ln"
    elif s.endswith("_prf"):
        result["g"] = GUIDANCE_STRATEGY__PERSISTENT_PREFIX_GUIDANCE
        suffix = "_prf"
    else:
        raise ValueError(f"Unknown guidance strategy suffix in: {s}")

    base = s[: -len(suffix)]

    if base.endswith("_lci"):
        result["prompt_type"] = PROMPT_TYPE__INSTRUCT_LONG_CODE_PROMPT
        base = base[: -len("_lci")]
    elif result["g"] == GUIDANCE_STRATEGY__TOKEN_GUIDANCE:
        result["prompt_type"] = PROMPT_TYPE__DEEPSEEK_INSTRUCT
    elif result["g"] == GUIDANCE_STRATEGY__LINE_GUIDANCE:
        result["prompt_type"] = PROMPT_TYPE__DEEPSEEK_BASE
    elif result["g"] == GUIDANCE_STRATEGY__PERSISTENT_PREFIX_GUIDANCE:
        result["prompt_type"] = PROMPT_TYPE__DEEPSEEK_INSTRUCT

    if "p" in base:
        result["p"] = True
        base = base.replace("p", "", 1)
    if "b" in base:
        result["b"] = True
        base = base.replace("b", "", 1)

    match = re.search(r"ns(\d+)t([\d.]+)d(\w+)", base)
    if match:
        result["n"] = True
        result["s"] = int(match.group(1))
        result["t"] = float(match.group(2))
        d_raw = match.group(3)
        result["d"] = d_raw if d_raw == "inf" else int(d_raw)

    return result


# --- Main ---
def main():
    if len(sys.argv) != 2:
        print("Usage: python verify_dynamic_signals_dir.py <directory>")
        sys.exit(1)

    input_dir = sys.argv[1]
    if not os.path.isdir(input_dir):
        print(f"[Error] '{input_dir}' is not a directory.")
        sys.exit(1)

    all_ok = True
    for name in os.listdir(input_dir):
        subdir_path = os.path.join(input_dir, name)
        if not os.path.isdir(subdir_path):
            continue

        try:
            parsed_args = parse_dynamic_signals_str(name)
            args_namespace = Namespace(**parsed_args)
            _, rebuilt = get_dynamic_signals(args_namespace)

            if name != rebuilt:
                print(f"X Mismatch: {name}  !=  {rebuilt}")
                all_ok = False
            else:
                print(f"V {name}")
        except Exception as e:
            print(f"[Error] Failed on '{name}': {e}")
            all_ok = False

    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
