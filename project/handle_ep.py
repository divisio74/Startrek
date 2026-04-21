##
## EPITECH PROJECT, 2026
## G-AIA-401-LYN-4-1-starttrek-7
## File description:
## handle_episode
##



def run_episode(env, policy):
    obs, info = env.reset()
    total_return = 0
    ep_len = 0
    running = 0
    while running != 1:
        action = policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        if (terminated or truncated == 1):
            running = 1
        total_return += reward
        ep_len = ep_len + 1
    return total_return , ep_len