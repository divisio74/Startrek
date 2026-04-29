##
## EPITECH PROJECT, 2026
## G-AIA-401-LYN-4-1-starttrek-7
## File description:
## eval
##

import argparse
import yaml
import gymnasium as gym
from handle_ep import run_episode
from policies import first_policy
from policies import epsilon_decay
from collections import Counter
import torch.nn as nn
from collections import deque # liste chaîné autogestionné
import torch
from replay import Replay

NB_INPUT = 8
NB_ACTION = 4
NB_HIDDEN = 128

def policy_handler(config, env, epsilon, actual_n):
    if config["policy"] == "random":
        return lambda obs: env.action_space.sample()
    elif config["policy"] == "epsilon":
        return lambda obs: epsilon_decay(obs, epsilon, env, actual_n)
    elif config["policy"] == "heuristic":
        return first_policy
    else:
        raise ValueError(f"Unknown policy: {config['policy']}")

def env_handler(config):
    product = config.get("video_folder")

    if product :
        render_mode = "rgb_array"
    else :
        render_mode = config.get("render")

    env = gym.make(config["env"], render_mode=render_mode)
    if product:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=product,
            episode_trigger=lambda i: True,
            name_prefix=config["policy"],
        )
    return env

def create_nn():
    n_n = nn.Sequential(
        nn.Linear(NB_INPUT, NB_HIDDEN), nn.ReLU(),
        nn.Linear(NB_HIDDEN, NB_HIDDEN), nn.ReLU(),
        nn.Linear(NB_HIDDEN, NB_ACTION)
    )
    return n_n

def run_eval(config):

    actual_n = create_nn()
    target_n = create_nn()
    optimizer = torch.optim.Adam(actual_n.parameters(), lr=1e-3)
    replay = Replay()
    target_n.load_state_dict(actual_n.state_dict())

    env = env_handler(config)
    epsilon = 1.0

    n = config["episodes"]
    returns = []
    lengths = []
    causes = []
    replay = Replay()
    step = 0

    for i in range(n):
        policy = policy_handler(config, env, epsilon, actual_n)
        total_return, ep_len, ending_cause, epsilon, step = run_episode(env, policy, replay, optimizer, target_n, actual_n, epsilon, step)
        returns.append(total_return)
        lengths.append(ep_len)
        causes.append(ending_cause)
        epsilon = max(epsilon * 0.995, 0.01)
        print(f"  episode number : {i+1}/{n}-> total_return = {total_return:.2f}, action_len = {ep_len}")
        print(f"{ending_cause}")


    mean_return = sum(returns) / n
    mean_action_len = sum(lengths) / n
    print(f"[{config['policy']}] mean_return ={ mean_return:.2f}, mean_action_len = {mean_action_len:.1f} avec {n} runs")
    print(f"causes: {dict(Counter(causes))}")

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    run_eval(cfg)
