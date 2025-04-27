import json
import argparse
import re
from collections import defaultdict
import pprint

MBPP_SIZE = 500


def load_entries(entries_path):
    """Load the input JSON containing entries with name and task_ids."""
    with open(entries_path, "r") as f:
        return json.load(f)


def dynamis_sigansl_str_to_cmdline_args(dynamic_signals_str):
    """Parse dynamic signal string into args."""
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
        args["g"] = "TOKEN_GUIDANCE"
        suffix = "_tok"
    elif dynamic_signals_str.endswith("_ln"):
        args["g"] = "LINE_GUIDANCE"
        suffix = "_ln"
    elif dynamic_signals_str.endswith("_prf"):
        args["g"] = "PERSISTENT_PREFIX_GUIDANCE"
        suffix = "_prf"
    else:
        raise ValueError(f"Unknown guidance strategy suffix in: {dynamic_signals_str}")

    base = dynamic_signals_str[: -len(suffix)]

    if base.endswith("_lci"):
        args["prompt_type"] = "PROMPT_TYPE__INSTRUCT_LONG_CODE_PROMPT"
        base = base[: -len("_lci")]
    if args["prompt_type"] is None:
        args["prompt_type"] = "PROMPT_TYPE__DEEPSEEK_INSTRUCT"

    if base.startswith("p"):
        args["p"] = True
        base = base.replace("p", "", 1)
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


def is_valid_args(args):
    """Check if args meet acceptance criteria."""
    is_valid = True
    is_valid &= args["g"] == "LINE_GUIDANCE"
    is_valid &= args["d"] != "inf"
    return is_valid


def compute_param_extension_cost(existing_params, new_args):
    """Compute how many new param values the new args would add."""
    cost = 0
    for key, value in new_args.items():
        if value not in existing_params[key]:
            cost += 1
    return cost


def param_priority_extension(entry, aggregation_params):
    """Tie-breaker priority: (t, d, s, prompt_type, name)."""
    args = dynamis_sigansl_str_to_cmdline_args(entry["name"])
    t_extends = 0 if args["t"] in aggregation_params["t"] else 1
    d_extends = 0 if args["d"] in aggregation_params["d"] else 1
    s_extends = 0 if args["s"] in aggregation_params["s"] else 1
    prompt_type_extends = (
        0 if args["prompt_type"] in aggregation_params["prompt_type"] else 1
    )
    return (t_extends, d_extends, s_extends, prompt_type_extends, entry["name"])


def greedy_aggregate(entries, threshold_ratio):
    """Main aggregation function."""
    threshold = int(threshold_ratio * MBPP_SIZE)

    covered_tasks = set()
    selected = []
    aggregation_params = defaultdict(set)

    remaining_entries = entries.copy()

    while len(covered_tasks) < threshold:
        best_gain = -1
        best_candidates = []

        for entry in remaining_entries:
            args = dynamis_sigansl_str_to_cmdline_args(entry["name"])
            if not is_valid_args(args):
                continue  # Skip invalid candidates

            task_ids = set(entry["task_ids"])
            new_tasks = task_ids - covered_tasks
            gain = len(new_tasks)

            if gain > best_gain:
                best_gain = gain
                best_candidates = [entry]
            elif gain == best_gain:
                best_candidates.append(entry)

        if not best_candidates:
            print("No more valid candidates to select from. Stopping.")
            break

        # Tie-break among best candidates: minimal param extension
        best_cost = float("inf")
        best_entries = []

        for entry in best_candidates:
            args = dynamis_sigansl_str_to_cmdline_args(entry["name"])
            cost = compute_param_extension_cost(aggregation_params, args)

            if cost < best_cost:
                best_cost = cost
                best_entries = [entry]
            elif cost == best_cost:
                best_entries.append(entry)

        if best_gain <= 0:
            print("No candidates can improve coverage. Stopping.")
            break

        # Final tie-break: prefer minimal (t, d, s, prompt_type), then name
        best_entry = min(
            best_entries, key=lambda e: param_priority_extension(e, aggregation_params)
        )

        # Add the best entry
        selected.append(best_entry)
        new_args = dynamis_sigansl_str_to_cmdline_args(best_entry["name"])

        for key, value in new_args.items():
            aggregation_params[key].add(value)

        covered_tasks.update(best_entry["task_ids"])
        remaining_entries.remove(best_entry)

        print(
            f"Selected {best_entry['name']} | Coverage: {len(covered_tasks)}/{MBPP_SIZE} ({len(covered_tasks)/MBPP_SIZE:.2%})"
        )

    # Build final aggregation output
    coverage_percentage = (len(covered_tasks) / MBPP_SIZE) * 100
    aggregation = {
        "coverage_percentage": coverage_percentage,
        "coverage_tasks_count": len(covered_tasks),
        "selected_names": [e["name"] for e in selected],
        "params": aggregation_params,
    }

    return aggregation


def compute_grid_size(aggregation):
    """Compute total grid size based on aggregation params."""
    params = aggregation["params"]

    d_options = len(params.get("d", []))
    s_options = len(params.get("s", []))
    t_options = len(params.get("t", []))
    prompt_type_options = len(params.get("prompt_type", []))

    total = d_options * s_options * t_options * prompt_type_options

    p_options = params.get("p", [False])
    if True in p_options:
        total += prompt_type_options
    return total


def summarize_aggregation(aggregation):
    """Summarize aggregation results: coverage, selected names, grid size, ratio."""
    coverage = aggregation.get("coverage_percentage", 0.0)
    coverage_tasks = aggregation.get("coverage_tasks_count", 0.0)
    names_len = len(aggregation.get("selected_names", []))
    grid_size = compute_grid_size(aggregation)
    perc = (names_len / grid_size) * 100 if names_len > 0 else 0.0

    print("\n=== Aggregation Summary ===")
    print(f"Coverage percentage: {coverage:.2f}% ({coverage_tasks} / 500)")
    coverage = aggregation.get("coverage_percentage", 0.0) + (257.0 / 500) * 100
    print(f"Total Coverage percentage: {coverage:.2f}% ({coverage_tasks + 257} / 500)")
    print(f"Selected names: {names_len}")
    print(f"Total grid size: {grid_size}")
    print(f"Grid / Selected names: {perc:.1f}%")
    print("============================\n")


def save_aggregation(aggregation, output_path):
    """Save aggregation output as JSON with sorted lists."""
    serializable_aggregation = {}
    serializable_aggregation["coverage_percentage"] = aggregation["coverage_percentage"]
    serializable_aggregation["selected_names"] = aggregation["selected_names"]

    serializable_aggregation["params"] = {}
    for key, values in aggregation["params"].items():
        if None in values:
            values = [v for v in values if v is not None]
        serializable_aggregation["params"][key] = sorted(list(values))

    with open(output_path, "w") as f:
        json.dump(serializable_aggregation, f, indent=2)
    pprint.pprint(serializable_aggregation)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Greedy dynamic signal aggregation with task_ids coverage goal."
    )
    parser.add_argument(
        "--entries",
        type=str,
        required=True,
        help="Path to JSON with entries (name + task_ids).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Coverage threshold ratio (default: 0.82 for 82%).",
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to output aggregation JSON."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    entries = load_entries(args.entries)

    aggregation = greedy_aggregate(entries, args.threshold)

    print(f"\nAggregation completed:")
    print(f"  Selected names: {len(aggregation['selected_names'])}")
    print(f"  Final coverage: {aggregation['coverage_percentage']}%")

    save_aggregation(aggregation, args.output)
    summarize_aggregation(aggregation)
    print(f"Aggregation saved to {args.output}")


if __name__ == "__main__":
    main()
