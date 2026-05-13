##
## EPITECH PROJECT, 2026
## G-AIA-401-LYN-4-1-starttrek-7
## File description:
## plot3
##

import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

WINDOW = 20

if len(sys.argv) < 2:
    print("usage: python plot3.py <train_log.csv> [<train_log.csv> ...]")
    sys.exit(1)

fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
ax_ret, ax_eps, ax_len = axes

for path in sys.argv[1:]:
    df = pd.read_csv(path)
    label = os.path.splitext(os.path.basename(path))[0]
    episodes = df["episode"]

    ax_ret.plot(episodes, df["return"], alpha=0.25)
    rolling = df["return"].rolling(WINDOW, min_periods=1).mean()
    line, = ax_ret.plot(episodes, rolling, label=f"{label} (MA{WINDOW})")
    color = line.get_color()

    ax_eps.plot(episodes, df["epsilon"], color=color, label=label)
    ax_len.plot(
        episodes,
        df["length"].rolling(WINDOW, min_periods=1).mean(),
        color=color,
        label=label,
    )

ax_ret.set_ylabel("Return")
ax_ret.set_title("Évolution du training")
ax_ret.grid(True, alpha=0.3)
ax_ret.legend(fontsize=8)

ax_eps.set_ylabel("Epsilon")
ax_eps.grid(True, alpha=0.3)

ax_len.set_ylabel(f"Episode length (MA{WINDOW})")
ax_len.set_xlabel("Episode")
ax_len.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("train_evolution.png", dpi=150, bbox_inches="tight")
print("saved train_evolution.png")

for path in sys.argv[1:]:
    df = pd.read_csv(path)
    label = os.path.splitext(os.path.basename(path))[0]
    causes = df["cause"].value_counts().to_dict()
    print(f"  {label}: {len(df)} episodes  |  "
          f"mean_return = {df['return'].mean():.2f}  |  "
          f"final_epsilon = {df['epsilon'].iloc[-1]:.4f}")
    print(f"    causes: {causes}")
