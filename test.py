##
## EPITECH PROJECT, 2026
## RL
## File description:
## main
##

import gymnasium as gym

env = gym.make("LunarLander-v3", render_mode ="rgb_array")
observation, info = env.reset(seed=42)
    
for i in range(10000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        observation, info = env.reset()

env.close() 