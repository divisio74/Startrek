##
## EPITECH PROJECT, 2026
## G-AIA-401-LYN-4-1-starttrek-7
## File description:
## eval
##

import argparse
import os
import random
from collections import Counter

import gymnasium as gym
import torch
import torch.nn as nn
import yaml
from handle_ep import run_episode
from logger import make_log_path, write_csv
from policies import epsilon_greedy, first_policy

NB_INPUT = 8
NB_ACTION = 4
NB_HIDDEN = 128

EPSILON_EVAL = 0.0


def create_nn():
    return nn.Sequential(
        nn.Linear(NB_INPUT, NB_HIDDEN), nn.ReLU(),
        nn.Linear(NB_HIDDEN, NB_HIDDEN), nn.ReLU(),
        nn.Linear(NB_HIDDEN, NB_ACTION),
    )


def env_handler(config):
    product = config.get("video_folder")
    render_mode = "rgb_array" if product else config.get("render")

    env = gym.make(config["env"], render_mode=render_mode)
    if product:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=product,
            episode_trigger=lambda i: True,
            name_prefix=config["policy"],
        )
    return env


def policy_handler(config, env, epsilon, actual_n):
    if config["policy"] == "random":
        return lambda obs: env.action_space.sample()
    elif config["policy"] == "epsilon":
        return lambda obs: epsilon_greedy(obs, epsilon, env, actual_n)
    elif config["policy"] == "heuristic":
        return first_policy
    else:
        raise ValueError(f"Unknown policy: {config['policy']}")


def run_eval(config):
    seed = config.get("seed")
    env = env_handler(config)
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        env.reset(seed=seed)
        env.action_space.seed(seed)

    actual_n = None
    if config["policy"] == "epsilon":
        actual_n = create_nn()
        model_path = config.get("model_path")
        if model_path:
            actual_n.load_state_dict(torch.load(model_path))
            print(f"model loaded from {model_path}")
        else:
            print("No model_path in config — using random weights")

    n = config["episodes"]
    returns = []
    lengths = []
    causes = []
    rows = []

    for i in range(n):
        policy = policy_handler(config, env, EPSILON_EVAL, actual_n)
        total_return, ep_len, ending_cause, _ = run_episode(env, policy, train=False)
        returns.append(total_return)
        lengths.append(ep_len)
        causes.append(ending_cause)
        rows.append({
            "episode": i + 1,
            "return": round(total_return, 4),
            "length": ep_len,
            "cause": ending_cause,
            "seed": seed,
        })
        print(f"  episode {i + 1}/{n} -> total_return = {total_return:.2f}, action_len = {ep_len}")
        print(f"  cause: {ending_cause}")

    mean_return = sum(returns) / n
    mean_action_len = sum(lengths) / n
    print(f"[eval | {config['policy']}] mean_return = {mean_return:.2f}, mean_action_len = {mean_action_len:.1f} avec {n} runs")
    print(f"causes: {dict(Counter(causes))}")

    modelpath = config.get("model_path")
    if modelpath:
        model_id = os.path.splitext(os.path.basename(modelpath))[0]
        log_path = make_log_path("eval", model_id)
    else:
        log_path = make_log_path("eval", config["policy"], seed)

    write_csv(log_path, rows, ["episode", "return", "length", "cause", "seed"])
    print(f"log -> {log_path}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    run_eval(cfg)
