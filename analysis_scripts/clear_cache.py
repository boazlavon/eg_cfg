import os
import json

root_dir = "/home/ai_center/ai_users/boazlavon/data/code/prod/dsgi/results/mbpp/deepseek-ai_deepseek-coder-1.3b-instruct"
for dir_name in os.listdir(root_dir):
    if not dir_name.endswith("lci_ln"):
        continue

    dir_path = os.path.join(root_dir, dir_name)

    if not os.path.isdir(dir_path):
        continue

    for file_name in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file_name)

        if not os.path.isfile(file_path):
            continue

        try:
            with open(file_path, "r") as f:
                content = f.read()
                if '"cached": true' in content:
                    print(f"Deleting {file_path}")
                    os.remove(file_path)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
