##
## EPITECH PROJECT, 2026
## G-AIA-401-LYN-4-1-starttrek-7
## File description:
## handle_episode
##

import torch
import torch.nn.functional as F
from classification import ending_info

GAMMA = 0.99
TARGET_UPDATE_EVERY = 100


def learn(batch, optimizer, target_n, actual_n):
    obs, actions, rewards, next_obs, dones = batch
    with torch.no_grad():
        next_s = target_n(next_obs).max(dim=1).values
        target = rewards + GAMMA * next_s * (1 - dones)
    predicted = actual_n(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
    loss = F.mse_loss(predicted, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def run_episode(
    env,
    policy,
    replay=None,
    optimizer=None,
    target_n=None,
    actual_n=None,
    step=0,
    train=False,
):
    obs_actual, _ = env.reset()
    total_return = 0
    ep_len = 0
    terminated = False
    truncated = False
    reward = 0

    while True:
        obs_old = obs_actual
        action = policy(obs_actual)
        obs_actual, reward, terminated, truncated, _ = env.step(action)

        if train:
            replay.push(obs_old, action, reward, obs_actual, terminated)
            step += 1
            if replay.ready():
                batch = replay.data_for_train()
                learn(batch, optimizer, target_n, actual_n)
            if step % TARGET_UPDATE_EVERY == 0:
                target_n.load_state_dict(actual_n.state_dict())

        total_return += reward
        ep_len += 1
        if terminated or truncated:
            break

    ending_cause = ending_info(terminated, truncated, obs_actual, reward)
    return total_return, ep_len, ending_cause, step
