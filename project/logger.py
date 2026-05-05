##
## EPITECH PROJECT, 2026
## G-AIA-401-LYN-4-1-starttrek-7
## File description:
## logger
##

import csv
import os
from datetime import datetime


def make_log_path(prefix, policy=None, seed=None, log_dir="logs"):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parts = [prefix]
    if policy is not None:
        parts.append(policy)
    if seed is not None:
        parts.append(f"seed{seed}")
    parts.append(timestamp)
    return os.path.join(log_dir, "_".join(parts) + ".csv")


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
