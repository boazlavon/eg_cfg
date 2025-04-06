import os
import argparse
import json
import ast
from difflib import SequenceMatcher
from collections import defaultdict
import zss


def load_json(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except:
        return None


def load_baseline_jsonl(baseline_path):
    generations = {}
    with open(baseline_path, "r") as f:
        for line in f:
            entry = json.loads(line)
            generations[entry["task_id"]] = entry["generation"]
    return generations


def compare_strings(str1, str2):
    return str1.strip() == str2.strip()


def string_similarity(str1, str2):
    return SequenceMatcher(None, str1.strip(), str2.strip()).ratio()


def normalize_ast(code):
    try:
        tree = ast.parse(code)
        return ast.dump(tree, annotate_fields=False, include_attributes=False)
    except Exception:
        return None


def structural_similarity(code1, code2):
    ast1 = normalize_ast(code1)
    ast2 = normalize_ast(code2)
    if ast1 is None or ast2 is None:
        return 0.0
    return SequenceMatcher(None, ast1, ast2).ratio()


def node_label(node):
    return type(node).__name__


def to_zss_tree(node):
    if not isinstance(node, ast.AST):
        return None
    children = [to_zss_tree(child) for child in ast.iter_child_nodes(node)]
    children = [child for child in children if child is not None]
    return zss.Node(node_label(node), children)


def tree_edit_distance(code1, code2):
    try:
        tree1 = ast.parse(code1)
        tree2 = ast.parse(code2)
        zss_tree1 = to_zss_tree(tree1)
        zss_tree2 = to_zss_tree(tree2)
        return zss.simple_distance(zss_tree1, zss_tree2)
    except Exception:
        return float("inf")


def normalized_tree_similarity(code1, code2):
    dist = tree_edit_distance(code1, code2)
    return 1 / (1 + dist)


def analyze_trial_against_baseline(
    trial_dir, baseline_generations, similarity_threshold=0.95, ast_threshold=0.7
):
    identical = []
    similar = []
    ast_similar = []
    different = []
    missing = []
    errors = []
    multiple_functions = []

    for task_id, baseline_gen in baseline_generations.items():
        fname = f"task_id={task_id}_gamma=0.0.json"
        fpath = os.path.join(trial_dir, fname)

        if not os.path.exists(fpath):
            missing.append(task_id)
            print("#" * 10)
            print(task_id)
            print("Official")
            print(baseline_gen)
            print("#" * 10)
            print()
            continue

        trial_data = load_json(fpath)
        if trial_data is None:
            missing.append(task_id)
            continue

        if trial_data.get("general_error") or trial_data.get("has_testcase_error"):
            errors.append(task_id)
            print("#" * 10)
            print(task_id)
            print("Official")
            print(baseline_gen)
            print("#" * 10)
            print()
            continue

        trial_gen = trial_data.get("code", "")
        def_count = 0
        if trial_gen:
            def_count = trial_gen.count("def")
        if def_count > 1:
            multiple_functions.append(task_id)
            print("=" * 10)
            print(task_id)
            print("Official")
            print(baseline_gen)
            print()
            print("My")
            print(trial_gen)
            print()
            print("=" * 10)
            print()

        if compare_strings(baseline_gen, trial_gen):
            identical.append(task_id)
        elif string_similarity(baseline_gen, trial_gen) >= similarity_threshold:
            similar.append(task_id)
        elif normalized_tree_similarity(baseline_gen, trial_gen) >= ast_threshold:
            ast_similar.append(task_id)
        else:
            different.append(task_id)
            print("=" * 10)
            print(task_id)
            print("Official")
            print(baseline_gen)
            print()
            print("My")
            print(trial_gen)
            print()
            print("=" * 10)
            print()

    total = len(baseline_generations)
    print(f"\n=== Comparison for trial: {trial_dir} ===")
    print(f"Total tasks in baseline: {total}")
    print(f"Identical: {len(identical)}")
    print(f"String-similar (>{similarity_threshold}): {len(similar)}")
    print(f"AST-similar (>{ast_threshold}): {len(ast_similar)}")
    print(f"Different: {len(different)}")
    print(different)
    print(f"Errors: {len(errors)}")
    print(errors)
    print(f"Missing: {len(missing)}")
    print(missing)
    print(f"Multiple Function: {len(multiple_functions)}")
    print(multiple_functions)

    return {
        "trial": trial_dir,
        "identical": identical,
        "similar": similar,
        "ast_similar": ast_similar,
        "different": different,
        "missing": missing,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare baseline generations to gamma=0.0 trial generations."
    )
    parser.add_argument(
        "--baseline", type=str, required=True, help="Path to baseline .jsonl file"
    )
    parser.add_argument(
        "--trials-dir",
        type=str,
        required=True,
        help="Path to directory containing trial subdirectories",
    )
    parser.add_argument(
        "--string-threshold",
        type=float,
        default=0.95,
        help="String similarity threshold",
    )
    parser.add_argument(
        "--ast-threshold", type=float, default=0.7, help="AST similarity threshold"
    )
    args = parser.parse_args()

    baseline_generations = load_baseline_jsonl(args.baseline)

    for trial_name in os.listdir(args.trials_dir):
        if "p" not in trial_name:
            continue
        trial_path = os.path.join(args.trials_dir, trial_name)
        if os.path.isdir(trial_path):
            analyze_trial_against_baseline(
                trial_path,
                baseline_generations,
                similarity_threshold=args.string_threshold,
                ast_threshold=args.ast_threshold,
            )


if __name__ == "__main__":
    main()
