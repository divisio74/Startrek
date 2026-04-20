##
## EPITECH PROJECT, 2026
## RL
## File description:
## main
##

import gymnasium as gym

env = gym.make("LunarLander-v3", render_mode ="rgb_array")
Machine_state, info = env.reset(seed=42)

for i in range(10000):
    action = env.action_space.sample()
    Machine_state, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        Machine_state, info = env.reset()


env.close()