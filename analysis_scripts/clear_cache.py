import os
from concurrent.futures import ProcessPoolExecutor, as_completed

root_dir = "results/mbpp/deepseek-ai_deepseek-coder-1.3b-instruct"
max_workers = 8

def process_directory(dir_path):
    print(dir_path)
    try:
        for file_name in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file_name)
            if not os.path.isfile(file_path):
                continue

            with open(file_path, "r") as f:
                content = f.read()
                if 'cached' in content:
                    print(f"Deleting {file_path}")
                    os.remove(file_path)
    except Exception as e:
        print(f"Error processing directory {dir_path}: {e}")

if __name__ == "__main__":
    dir_paths = [
        os.path.join(root_dir, dir_name)
        for dir_name in os.listdir(root_dir)
        # if dir_name.endswith("lci_ln") and os.path.isdir(os.path.join(root_dir, dir_name))
    ]
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_directory, path) for path in dir_paths]
        for future in as_completed(futures):
            future.result()  # to raise any exceptions
