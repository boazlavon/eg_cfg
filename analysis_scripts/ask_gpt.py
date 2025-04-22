import json
import os
from datasets import load_dataset
from openai import OpenAI

api_key = "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"
client = OpenAI(api_key=api_key)

test_ds = load_dataset("google-research-datasets/mbpp", "full", split="test")

OUTPUT_DIR = "analysis"


def validate_task_solutions(task):
    task_id = task["task_id"]
    task_json = json.dumps(task, indent=4)
    gamma_0_file = os.path.join(OUTPUT_DIR, f"task_id={task_id}_gamma=0.0.json")
    code_analysis_file = os.path.join(
        OUTPUT_DIR, f"indexed/indexed_code_analysis_{task_id}.json"
    )

    with open(gamma_0_file, "r") as f:
        gamma_0_solution = json.load(f)["code"]

    # Check gamma=0 solution
    prompt_gamma_0 = (
        f"MBPP Task:\n{task_json}\n\n"
        f"Given Solution (gamma=0):\n{gamma_0_solution}\n\n"
        "Evaluate the solution in three ways:\n"
        "1. How well it aligns with the task description.\n"
        "2. How well it aligns with the provided unit tests.\n"
        "3. Whether the task description or tests contain internal inconsistencies or errors.\n\n"
        'Respond in JSON format: {"alignment_with_description": {"score": float (0 to 1), "explanation": str}, "alignment_with_tests": {"score": float (0 to 1), "explanation": str}, "task_issue": {"error_present": true/false, "explanation": str}}'
    )

    response_gamma_0 = client.chat.completions.create(
        model="gpt-4o-mini", messages=[{"role": "user", "content": prompt_gamma_0}]
    )

    gamma_0_result = response_gamma_0.choices[0].message.content
    gamma_0_output_file = os.path.join(OUTPUT_DIR, f"gamma_0_validation_{task_id}.json")
    with open(gamma_0_output_file, "w") as f:
        f.write(gamma_0_result)
    print(gamma_0_result)

    print(f"Gamma=0 Solution validation saved to {gamma_0_output_file}")

    # Upload generated solutions file to avoid token limit
    import ipdb

    ipdb.set_trace()
    file_upload = client.files.create(
        file=open(code_analysis_file, "rb"), purpose="assistants"
    )

    assistant = client.beta.assistants.create(
        name="Solution Validator",
        instructions="You're validating code solutions for MBPP benchmark problems.",
        tools=[{"type": "file_search"}],
        model="gpt-4o",
    )

    thread = client.beta.threads.create()

    client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=[
            {
                "type": "text",
                "text": (
                    f"MBPP Task:\n{task_json}\n\n"
                    "The uploaded file contains indexed code solutions.\n\n"
                    "Please iterate over each solution. For each one, evaluate in three ways:\n"
                    "1. How well it aligns with the task description.\n"
                    "2. How well it aligns with the unit tests.\n"
                    "3. Whether the task description or tests contain internal inconsistencies or errors.\n\n"
                    'Respond in JSON format as: {"solutions": [\n  {\n    "index": int,\n    "alignment_with_description": {"score": float (0 to 1), "explanation": str},\n    "alignment_with_tests": {"score": float (0 to 1), "explanation": str},\n    "task_issue": {"error_present": true/false, "explanation": str}\n  }, ...]}'
                ),
            },
        ],
    )
    # file_ids=[file_upload.id]
    import ipdb

    ipdb.set_trace()
    # run = client.beta.threads.runs.create(thread_id=thread.id, assistant_id=assistant.id)
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id,
        file_ids=[file_upload.id],  # ✅ Attach file at run level
    )

    print("Waiting for assistant to complete run...")
    while True:
        run_status = client.beta.threads.runs.retrieve(
            thread_id=thread.id, run_id=run.id
        )
        if run_status.status == "completed":
            break

    messages = client.beta.threads.messages.list(thread_id=thread.id)
    final_response = messages.data[0].content[0].text.value

    generated_output_file = os.path.join(
        OUTPUT_DIR, f"generated_solutions_validation_{task_id}.json"
    )
    with open(generated_output_file, "w") as f:
        f.write(final_response)

    print(f"Generated Solutions validation saved to {generated_output_file}")


JSON_FILE = "not_solved.json"
with open(JSON_FILE, "r") as f:
    tasks = json.load(f)

tasks = tasks[:2]
for task in tasks:
    import ipdb

    ipdb.set_trace()
    task_id = task["task_id"]
    print(f"task_id={task_id}")
    validate_task_solutions(task)
