import json
import sys
import re
import argparse
from pathlib import Path

TASK_ID_RE = re.compile(r"task_id=(.+?)_gamma=")


def find_passed_task_ids(json_dir):
    json_dir = Path(json_dir)
    for path in json_dir.rglob("*.json"):
        match = TASK_ID_RE.search(path.name)
        if not match:
            continue  # skip files that don't match the pattern
        task_id = match.group(1)
        try:
            with open(path) as f:
                data = json.load(f)
                if isinstance(data, dict) and data.get("passed") is True:
                    print(task_id)
        except Exception as e:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("json_dir", help="Directory containing JSON files")
    args = parser.parse_args()
    find_passed_task_ids(args.json_dir)
