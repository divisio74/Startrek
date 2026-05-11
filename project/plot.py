##
## EPITECH PROJECT, 2026
## G-AIA-401-LYN-4-1-starttrek-7
## File description:
## plot
##

import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ORDER = {"1K": 0, "10K": 1, "50K": 2, "90K": 3}

groups = {}
for path in sys.argv[1:]:
    df = pd.read_csv(path)
    m = re.search(r"test_(\d+K)", os.path.basename(path))
    if not m:
        continue
    bs = m.group(1)
    groups.setdefault(bs, []).append(df["return"].rolling(20).mean().reset_index(drop=True))

for bs in sorted(groups.keys(), key=lambda x: ORDER.get(x, 99)):
    runs = groups[bs]
    data = pd.concat(runs, axis=1)
    mean = data.mean(axis=1)
    episodes = range(1, len(mean) + 1)
    line, = plt.plot(episodes, mean, label=f"buffer={bs}")
    if data.shape[1] > 1:
        ci = 1.96 * data.std(axis=1) / np.sqrt(data.shape[1])
        plt.fill_between(episodes, mean - ci, mean + ci, color=line.get_color(), alpha=0.2)

plt.xlabel("Episode")
plt.ylabel("Return")
plt.legend()
plt.savefig("ablation.png")
print("saved ablation.png")
