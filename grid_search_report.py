import os
from sklearn.model_selection import ParameterGrid
from collections import defaultdict
from pathlib import Path

# Configuration
TRIALS_DIR = "/home/ai_center/ai_users/boazlavon/data/code/prod/dsgi/results/mbpp/deepseek-ai_deepseek-coder-1.3b-instruct"  # Replace with the actual directory
# TEMP_VALUES = [0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]
TEMP_VALUES = [0.5,0.7,0.75,0.8,0.85,0.9,0.95]
S_VALUES = [2, 3, 4, 5, 6]
D_VALUES = [1, 2, 3, float('inf')]

# Define expected grid
grid = ParameterGrid({
    "t": TEMP_VALUES,
    "s": S_VALUES,
    "d": D_VALUES
})

# Collect existing directories
existing_dirs = set()
for entry in os.listdir(TRIALS_DIR):
    if not os.path.isdir(os.path.join(TRIALS_DIR, entry)):
        continue
    if entry.startswith("ns") and "t" in entry and "_ln" in entry:
        try:
            s_part = entry.split("ns")[1].split("t")[0]
            t_part = entry.split("t")[1].split("d")[0]
            d_part = entry.split("d")[1].split("_")[0]
            s = int(s_part)
            t = float(t_part)
            #d = int(d_part)
            d_raw = entry.split("d")[1].split("_")[0]
            d = float("inf") if d_raw == "inf" else int(d_raw)
            existing_dirs.add((s, t, d))
        except Exception:
            continue

# Analyze missing and existing
all_configs = {(config["s"], config["t"], config["d"]) for config in grid}
missing = sorted(all_configs - existing_dirs)
present = sorted(all_configs & existing_dirs)

# Report
print("===== Trail Existence Report =====")
print(f"Total expected: {len(all_configs)}")
print(f"Existing:       {len(present)}")
print(f"Missing:        {len(missing)}")
print("\nExamples of missing trails:")
for i, (s, t, d) in enumerate(missing[:10]):
    print(f"  ns{s}t{t}d{d}_ln")
if len(missing) > 10:
    print(f"...and {len(missing) - 10} more.")

# Optional: Save missing as file
missing_file = Path(TRIALS_DIR) / "missing_trials.txt"
with open(missing_file, "w") as f:
    for s, t, d in missing:
        f.write(f"ns{s}t{t}d{d}_ln\n")
print(f"\nSaved list of missing trials to {missing_file}")

import matplotlib.pyplot as plt
import numpy as np

# Prepare data for plotting
def make_grid_matrix(s_vals, t_vals, existing_set, d_val):
    matrix = np.zeros((len(s_vals), len(t_vals)))
    for i, s in enumerate(s_vals):
        for j, t in enumerate(t_vals):
            matrix[i, j] = (s, t, d_val) in existing_set
    return matrix

fig, axes = plt.subplots(1, len(D_VALUES), figsize=(5 * len(D_VALUES), 5), sharey=True)

if len(D_VALUES) == 1:
    axes = [axes]

for idx, d in enumerate(D_VALUES):
    ax = axes[idx]
    grid_matrix = make_grid_matrix(S_VALUES, TEMP_VALUES, existing_dirs, d)

    im = ax.imshow(grid_matrix, cmap="Greens", interpolation="nearest", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(TEMP_VALUES)))
    ax.set_yticks(np.arange(len(S_VALUES)))
    ax.set_xticklabels([str(t) for t in TEMP_VALUES])
    ax.set_yticklabels([str(s) for s in S_VALUES])
    ax.set_xlabel("Temperature (t)")
    if idx == 0:
        ax.set_ylabel("Seed (s)")
    d_label = "inf" if d == float("inf") else str(d)
    ax.set_title(f"d = {d_label}")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

fig.suptitle("Trail Existence Grid (Green = Exists, Black = Missing)", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
output_path = os.path.join(TRIALS_DIR, "grid_search_report.png")
plt.savefig(output_path)
print(f"\nSaved grid search visualization to {output_path}")

# plt.show()
