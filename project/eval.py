##
## EPITECH PROJECT, 2026
## G-AIA-401-LYN-4-1-starttrek-7
## File description:
## eval
##

import argparse
import json
import gymnasium as gym
from handle_ep import run_episode
from policies import first_policy

def run_eval(config):
    render_mode = config.get("render")
    env = gym.make(config["env"], render_mode=render_mode)


    if config["policy"] == "heuristic":
        policy = first_policy
    else:
        raise ValueError(f"Unknown policy: {config['policy']}")

    total_return, ep_len = run_episode(env, policy) #run d'un seule ep pour test
    print(f"[{config['policy']}] return={total_return:.2f}, length={ep_len}")
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = json.load(f)
    run_eval(cfg)
