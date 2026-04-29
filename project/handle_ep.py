##
## EPITECH PROJECT, 2026
## G-AIA-401-LYN-4-1-starttrek-7
## File description:
## handle_episode
##

from classification import ending_info
from train import learn

def run_episode(env, policy, replay, optimizer, target_n, actual_n, epsilon, step):
    obs_actual, info = env.reset()
    total_return = 0
    ep_len = 0
    running = 0
    while running != 1:
        obs_old = obs_actual
        action = policy(obs_actual)
        obs_actual, reward, terminated, truncated, info = env.step(action)
        replay.push(obs_old, action, reward, obs_actual, terminated)
        step += 1

        if replay.ready():
            batch = replay.data_for_train()
            learn(batch, optimizer, target_n, actual_n)
        if step % 100 == 0:
            target_n.load_state_dict(actual_n.state_dict())
        if (terminated or truncated == 1):
            running = 1
        total_return += reward
        ep_len = ep_len + 1

    ending_cause = ending_info(terminated, truncated, obs_actual, reward)
    return total_return, ep_len, ending_cause, epsilon, step
