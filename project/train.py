##
## EPITECH PROJECT, 2026
## G-AIA-401-LYN-4-1-starttrek-7
## File description:
## train
##

import argparse
import os
import random
from collections import Counter

import torch
import yaml

from eval import create_nn, env_handler, policy_handler
from handle_ep import run_episode
from logger import make_log_path, write_csv
from replay import Replay

LEARNING_RATE = 1e-3
EPSILON_START = 1.0
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.01


def run_train(config):
    seed = config.get("seed")
    test_id = config.get("test_id")
    buffer_size = config.get("buffer_size")
    env = env_handler(config)
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        env.reset(seed=seed)
        env.action_space.seed(seed)
    if buffer_size is not None:
        replay = Replay(buffer_size)
    else :
        buffer_size = 50000
        replay = Replay(buffer_size)



    actual_n = create_nn()
    target_n = create_nn()
    target_n.load_state_dict(actual_n.state_dict())
    optimizer = torch.optim.Adam(actual_n.parameters(), lr=LEARNING_RATE)

    epsilon = EPSILON_START
    step = 0
    n = config["episodes"]
    returns = []
    lengths = []
    causes = []
    rows = []

    for i in range(n):
        policy = policy_handler(config, env, epsilon, actual_n)
        total_return, ep_len, ending_cause, step = run_episode(
            env,
            policy,
            replay=replay,
            optimizer=optimizer,
            target_n=target_n,
            actual_n=actual_n,
            step=step,
            train=True,
        )
        returns.append(total_return)
        lengths.append(ep_len)
        causes.append(ending_cause)
        rows.append({
            "episode": i + 1,
            "return": round(total_return, 4),
            "length": ep_len,
            "cause": ending_cause,
            "epsilon": round(epsilon, 4),
            "seed": seed,
            "buffer_size": buffer_size,
        })
        epsilon = max(epsilon * EPSILON_DECAY, EPSILON_MIN)
        print(f"  episode {i + 1}/{n} -> total_return = {total_return:.2f}, action_len = {ep_len}")
        print(f"  cause: {ending_cause}")

    mean_return = sum(returns) / n
    mean_action_len = sum(lengths) / n
    print(f"[train | {config['policy']}] mean_return = {mean_return:.2f}, mean_action_len = {mean_action_len:.1f} avec {n} runs")
    print(f"causes: {dict(Counter(causes))}")

    os.makedirs("models", exist_ok=True)
    if test_id is not None:
        prefixe = f"Train{test_id}"
        model_path = f"models/{config['policy']}_seed{seed}_{test_id}.pt"
    else :
        prefixe = f"Train"
        model_path = f"models/{config['policy']}_seed{seed}_{buffer_size}.pt"
    torch.save(actual_n.state_dict(), model_path)
    print(f"model -> {model_path}")
    log_path = make_log_path(prefixe, config["policy"], seed)
    write_csv(log_path, rows, ["episode", "return", "length", "cause", "epsilon", "seed", "buffer_size"])
    print(f"log -> {log_path}")

    env.close()
    return actual_n


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    run_train(cfg)
