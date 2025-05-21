import json
import sys

if len(sys.argv) != 4:
    print("Usage: python merge_lists.py file1.json file2.json output.json")
    sys.exit(1)

file1, file2, output_file = sys.argv[1], sys.argv[2], sys.argv[3]

# Load the lists
with open(file1, "r") as f1:
    list1 = json.load(f1)

with open(file2, "r") as f2:
    list2 = json.load(f2)

# Merge, deduplicate, and sort
merged_list = sorted(set(list1 + list2))

# Save the result
with open(output_file, "w") as f_out:
    json.dump(merged_list, f_out, indent=2)

print(f"Merged list saved to {output_file} with {len(merged_list)} unique elements.")
