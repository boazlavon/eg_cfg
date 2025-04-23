import argparse
import re
from argparse import Namespace
from consts import *


def get_dynamic_signals_str(session_config, guidance_config, dynamic_signals_config):
    dynamic_signals_str = []
    prompt_type = None
    guidance_strategy = None

    if dynamic_signals_config[DYNAMIC_SIGNAL__PARTIAL_EXECUTION].is_enabled:
        dynamic_signals_str.append("p")
    if dynamic_signals_config[DYNAMIC_SIGNAL__NEAREST_FUTURE_EXECUTION].is_enabled:
        if (
            dynamic_signals_config[
                DYNAMIC_SIGNAL__NEAREST_FUTURE_EXECUTION
            ].nf_samples_depth
            is not None
        ):
            d_arg = str(
                dynamic_signals_config[
                    DYNAMIC_SIGNAL__NEAREST_FUTURE_EXECUTION
                ].nf_samples_depth
            )
        else:
            d_arg = "inf"
        s = dynamic_signals_config[
            DYNAMIC_SIGNAL__NEAREST_FUTURE_EXECUTION
        ].nf_samples_count
        t = dynamic_signals_config[DYNAMIC_SIGNAL__NEAREST_FUTURE_EXECUTION].temperature
        dynamic_signals_str.append(f"ns{s}t{t}d{d_arg}")
    if dynamic_signals_config[DYNAMIC_SIGNAL__BACKWARD].is_enabled:
        dynamic_signals_str.append("b")
    dynamic_signals_str = "".join(dynamic_signals_str)

    if session_config.prompt_type == PROMPT_TYPE__INSTRUCT_LONG_CODE_PROMPT:
        prompt_type = "lci"
        dynamic_signals_str = f"{dynamic_signals_str}_{prompt_type}"

    if guidance_config.guidance_strategy == GUIDANCE_STRATEGY__TOKEN_GUIDANCE:
        guidance_strategy = "tok"
    if guidance_config.guidance_strategy == GUIDANCE_STRATEGY__LINE_GUIDANCE:
        guidance_strategy = "ln"
    if (
        guidance_config.guidance_strategy
        == GUIDANCE_STRATEGY__PERSISTENT_PREFIX_GUIDANCE
    ):
        guidance_strategy = "prf"
    dynamic_signals_str = f"{dynamic_signals_str}_{guidance_strategy}"
    return dynamic_signals_str


def dynamis_sigansl_str_to_cmdline_args(dynamic_signals_str):
    args = {
        "prompt_type": None,
        "g": None,
        "p": False,
        "n": False,
        "d": None,
        "s": None,
        "t": None,
        "b": False,
    }

    if dynamic_signals_str.endswith("_tok"):
        args["g"] = GUIDANCE_STRATEGY__TOKEN_GUIDANCE
        suffix = "_tok"
    elif dynamic_signals_str.endswith("_ln"):
        args["g"] = GUIDANCE_STRATEGY__LINE_GUIDANCE
        suffix = "_ln"
    elif dynamic_signals_str.endswith("_prf"):
        args["g"] = GUIDANCE_STRATEGY__PERSISTENT_PREFIX_GUIDANCE
        suffix = "_prf"
    else:
        raise ValueError(f"Unknown guidance strategy suffix in: {dynamic_signals_str}")

    base = dynamic_signals_str[: -len(suffix)]

    if base.endswith("_lci"):
        args["prompt_type"] = PROMPT_TYPE__INSTRUCT_LONG_CODE_PROMPT
        base = base[: -len("_lci")]

    if base.startswith("p"):
        args["p"] = True
        base = base.replace("p", "")
    if "b" in base:
        args["b"] = True
        base = base.replace("b", "")

    match = re.search(r"ns(\d+)t([\d.]+)d(\w+)", base)
    if match:
        args["n"] = True
        args["s"] = int(match.group(1))
        args["t"] = float(match.group(2))
        d_raw = match.group(3)
        args["d"] = d_raw if d_raw == "inf" else int(d_raw)

    return args


def build_dsgi_session_manager_config(args):
    dynamic_signals_config = {
        DYNAMIC_SIGNAL__NEAREST_FUTURE_EXECUTION: Namespace(
            **{
                "is_enabled": bool(args.n),
                "temperature": args.t,
                "nf_samples_count": args.s,
                "nf_samples_depth": args.d,
            }
        ),
        DYNAMIC_SIGNAL__PARTIAL_EXECUTION: Namespace(
            **{
                "is_enabled": bool(args.p),
            }
        ),
        DYNAMIC_SIGNAL__BACKWARD: Namespace(**{"is_enabled": False}),
    }
    guidance_config = {
        "guidance_strategy": args.g,
        "retries_count": args.r,
        "gammas": GAMMAS,
    }
    guidance_config = Namespace(**guidance_config)

    session_config = {
        "model_name": args.model_name,
        "prompt_type": args.prompt_type,
        "is_prod": args.prod,
        "use_cache": args.cache,
    }
    session_config = Namespace(**session_config)
    return session_config, guidance_config, dynamic_signals_config


def get_cmdline_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, help="Name of the model used")
    # parser.add_argument("--b", action="store_true", help="Enable backward signal")
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
    parser.add_argument("--s", type=int, default=2, help="nf samples count")
    parser.add_argument("--t", type=float, default=0.1, help="nf temp")
    parser.add_argument(
        "--r", type=int, default=1, help="Attempts Count for each gamma (retries)"
    )
    # parser.add_argument("--d", type=int, default=3, help="Max Lines for nearest future (deepness)")
    parser.add_argument("--d", type=int, default=None, help="nf samples depth")
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
        parsed_args = dynamis_sigansl_str_to_cmdline_args(args.from_string)
        parsed_args["model_name"] = args.model_name
        parsed_args["prompt_type"] = args.prompt_type
        parsed_args["prod"] = args.prod
        parsed_args["cache"] = args.cache
        parsed_args["r"] = args.r
        args2 = Namespace(**parsed_args)
        args = args2
    return args
