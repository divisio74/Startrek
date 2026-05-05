##
## EPITECH PROJECT, 2026
## G-AIA-401-LYN-4-1-starttrek-7
## File description:
## plot
##

import sys

import matplotlib.pyplot as plt
import pandas as pd

for path in sys.argv[1:]:
    df = pd.read_csv(path)
    label = f"buffer={int(df['buffer_size'].iloc[0])}"
    plt.plot(df["episode"], df["return"].rolling(20).mean(), label=label)

plt.xlabel("Episode")
plt.ylabel("Return")
plt.legend()
plt.savefig("ablation.png")
print("saved ablation.png")
