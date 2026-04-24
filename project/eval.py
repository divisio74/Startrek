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
from collections import Counter


def run_eval(config):
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

    if config["policy"] == "random":
        policy = lambda obs: env.action_space.sample()
    elif config["policy"] == "heuristic":
        policy = first_policy
    else:
        raise ValueError(f"Unknown policy: {config['policy']}")

    n = config["episodes"]
    returns = []
    lengths = []
    causes = []

    for i in range(n):
        total_return, ep_len, ending_cause = run_episode(env, policy)
        returns.append(total_return)
        lengths.append(ep_len)
        causes.append(ending_cause)
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
