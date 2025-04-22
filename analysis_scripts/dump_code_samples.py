import json
import subprocess
import os
from multiprocessing import Pool

JSON_FILE = "not_solved.json"
OUTPUT_DIR = "analysis"
SCRIPT_PATH = "load_samples.py"
ROOT_DIR = "results/mbpp/deepseek-ai_deepseek-coder-1.3b-instruct/"
NUM_PROCESSES = 8  # Adjust based on your system

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the JSON file
with open(JSON_FILE, "r") as f:
    tasks = json.load(f)


def process_task(task):
    task_id = task["task_id"]
    print(f"task_id={task_id}")
    output_file = os.path.join(OUTPUT_DIR, f"generated_samples_{task_id}.txt")

    # Run the script and redirect stdout to the output file
    with open(output_file, "w") as outfile:
        subprocess.run(["python", SCRIPT_PATH, ROOT_DIR, str(task_id)], stdout=outfile)
        gamma_0_path = f"/a/home/cc/students/cs/boazlavon/code/prod/dsgi/results/mbpp/deepseek-ai_deepseek-coder-1.3b-instruct/ns2t0.9d8_ln/task_id={task_id}_gamma=0.0.json"
        subprocess.run(
            [
                "cp",
                "-v",
                gamma_0_path,
                f"analysis/indexed/indexed_code_analysis_{task_id}g0.json",
            ]
        )

    print(f"Processed task_id={task_id}, output written to {output_file}")


# Use multiprocessing to process tasks in parallel
with Pool(NUM_PROCESSES) as pool:
    pool.map(process_task, tasks)
