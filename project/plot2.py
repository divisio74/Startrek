##
## EPITECH PROJECT, 2026
## G-AIA-401-LYN-4-1-starttrek-7
## File description:
## plot2
##

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

returns_per_seed = {}
for path in sys.argv[1:]:
    df = pd.read_csv(path)
    seed = int(df["seed"].iloc[0])
    returns_per_seed[seed] = df["return"].values

seeds = sorted(returns_per_seed.keys())
data = [returns_per_seed[s] for s in seeds]
means = [np.mean(d) for d in data]

global_mean = np.mean(means)
std = np.std(means, ddof=1)
ci = 2.776 * std / np.sqrt(len(means))

cv = 100 * std / global_mean

plt.boxplot(data, labels=[f"seed {s}" for s in seeds], showmeans=True)
plt.axhline(global_mean, color="red", linestyle="--",
            label=f"mean = {global_mean:.1f} ± {ci:.1f} (95% CI)  |  CV inter-seed = {cv:.2f}%")

plt.ylabel("Return")
plt.title(f"Variation entre seeds (n={len(seeds)})")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("seeds_variation.png", dpi=150, bbox_inches="tight")
print("saved seeds_variation.png")
for s, d in zip(seeds, data, strict=True):
    print(f"  seed {s}: mean = {np.mean(d):.2f}  std = {np.std(d, ddof=1):.2f}  "
          f"min = {np.min(d):.1f}  max = {np.max(d):.1f}")
print(f"  GLOBAL: {global_mean:.2f} ± {ci:.2f} (95% CI)  "
      f"std_inter_seed = {std:.2f}")
