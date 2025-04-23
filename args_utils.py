import argparse
import re
from argparse import Namespace
import os
from consts import *


def get_dynamic_signals(args):
    dynamic_signals_str = []
    dynamic_signals_types = []
    prompt_type = None
    guidance_strategy = None

    if args.p:
        dynamic_signals_str.append("p")
        dynamic_signals_types.append(DYNAMIC_SIGNAL__PARTIAL_EXECUTION)
    if args.n:
        if args.d is not None:
            d_arg = str(args.d)
        else:
            d_arg = "inf"
        dynamic_signals_str.append(f"ns{args.s}t{args.t}d{d_arg}")
        dynamic_signals_types.append(DYNAMIC_SIGNAL__NEAREST_FUTURE_EXECUTION)
    if args.b:
        dynamic_signals_str.append("b")
        dynamic_signals_types.append(DYNAMIC_SIGNAL__BACKWARD)
    dynamic_signals_str = "".join(dynamic_signals_str)

    if args.prompt_type == PROMPT_TYPE__INSTRUCT_LONG_CODE_PROMPT:
        prompt_type = "lci"
        dynamic_signals_str = f"{dynamic_signals_str}_{prompt_type}"

    if args.g == GUIDANCE_STRATEGY__TOKEN_GUIDANCE:
        guidance_strategy = "tok"
    if args.g == GUIDANCE_STRATEGY__LINE_GUIDANCE:
        guidance_strategy = "ln"
    if args.g == GUIDANCE_STRATEGY__PERSISTENT_PREFIX_GUIDANCE:
        guidance_strategy = "prf"
    dynamic_signals_str = f"{dynamic_signals_str}_{guidance_strategy}"

    dynamic_signals_types = tuple(dynamic_signals_types)
    assert dynamic_signals_types
    return dynamic_signals_types, dynamic_signals_str


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

    if base.startswith("p"):
        result["p"] = True
        base = base.replace("p", "")
    if "b" in base:
        result["b"] = True
        base = base.replace("b", "")

    match = re.search(r"ns(\d+)t([\d.]+)d(\w+)", base)
    if match:
        result["n"] = True
        result["s"] = int(match.group(1))
        result["t"] = float(match.group(2))
        d_raw = match.group(3)
        result["d"] = d_raw if d_raw == "inf" else int(d_raw)

    return result


def create_result_dir(args):
    _, dynamic_signals_str = get_dynamic_signals(args)
    results_dir = os.path.join(
        "results", "mbpp", args.model_name.replace("/", "_"), dynamic_signals_str
    )
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def build_dsgi_session_manager_args(args):
    dynamic_signals_types, _ = get_dynamic_signals(args)
    results_dir = create_result_dir(args)
    dynamic_signals_args = {
        DYNAMIC_SIGNAL__NEAREST_FUTURE_EXECUTION: Namespace(
            **{
                "is_enabled": DYNAMIC_SIGNAL__NEAREST_FUTURE_EXECUTION
                in dynamic_signals_types,
                "temperature": args.t,
                "nf_samples_count": args.s,
                "max_lines": args.d,
            }
        ),
        DYNAMIC_SIGNAL__PARTIAL_EXECUTION: Namespace(
            **{
                "is_enabled": DYNAMIC_SIGNAL__PARTIAL_EXECUTION
                in dynamic_signals_types,
            }
        ),
    }

    guidance_args = {
        "guidance_strategy": args.g,
        "retries_count": args.r,
        "gammas": GAMMAS,
        "dynamic_signals_types": dynamic_signals_types,
    }
    guidance_args = Namespace(**guidance_args)

    session_args = {
        "model_name": args.model_name,
        "prompt_type": args.prompt_type,
        "results_dir": results_dir,
        "is_prod": args.prod,
        "use_cache": args.cache,
        "problem_start_idx": None,
        "problem_end_idx": None,
    }
    session_args = Namespace(**session_args)

    dsgi_session_manager_args = {
        "session_args": session_args,
        "guidance_args": guidance_args,
        "dynamic_signals_args": dynamic_signals_args,
    }
    return dsgi_session_manager_args


def get_cmdline_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, help="Name of the model used")
    parser.add_argument("--b", action="store_true", help="Enable backward signal")
    parser.add_argument("--n", action="store_true", help="Enable nearest future signal")
    parser.add_argument(
        "--p",
        action="store_true",
        help="Enable partial execution signal",
    )
    parser.add_argument(
        "--prompt-type",
        type=str,
        required=True,
        choices=VALID_PROMPT_TYPES,
        help="Type of prompt to use. Must be one of: " + ", ".join(VALID_PROMPT_TYPES),
    )
    parser.add_argument("--prod", action="store_true")
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--s", type=int, default=2, help="Nearest Future Sequences")
    parser.add_argument("--t", type=float, default=0.1, help="Temp")
    parser.add_argument(
        "--r", type=int, default=1, help="Attempts Count for each gamma (retries)"
    )
    # parser.add_argument("--d", type=int, default=3, help="Max Lines for nearest future (deepness)")
    parser.add_argument(
        "--d", type=int, default=None, help="Max Lines for nearest future (deepness)"
    )
    parser.add_argument(
        "--g",
        "--guidance",
        choices=GUIDANCE_STRATEGIES,
        default=GUIDANCE_STRATEGY__LINE_GUIDANCE,
        help="Guidance strategy to use. Options: %(choices)s (default: %(default)s)",
    )
    parser.add_argument("--from-string", default=None)
    args = parser.parse_args()
    if args.from_string:
        parsed_args = parse_dynamic_signals_str(args.from_string)
        parsed_args["model_name"] = args.model_name
        parsed_args["prompt_type"] = args.prompt_type
        parsed_args["prod"] = args.prod
        parsed_args["cache"] = args.cache
        parsed_args["r"] = args.r
        args2 = Namespace(**parsed_args)
        args = args2
    return args
