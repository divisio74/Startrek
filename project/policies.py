##
## EPITECH PROJECT, 2026
## G-AIA-401-LYN-4-1-starttrek-7
## File description:
## policies
##

import random
import torch

def epsilon_decay(alunisseur, epsilon, env, o_net):
    if random.random() < epsilon:
        return env.action_space.sample()  # exploration
    else:
        obs_tensor = torch.tensor(alunisseur, dtype=torch.float32)
        return o_net(obs_tensor).argmax().item()

def first_policy(obs):
    angle = obs[4]
    vitesse_verti = obs[3]
    pos_horiz = obs[0]

    if angle > 0.1 or pos_horiz < -0.2 :
        return 3
    elif angle < -0.1 or pos_horiz > 0.2:
        return 1
    elif vitesse_verti < -0.5:
        return 2
    else:
        return 0

def first_policyV2(obs):
    angle = obs[4]
    vitesse_verti = obs[3]
    pos_horiz = obs[0]

    if angle > 0.1 or pos_horiz < -0.2 :
        return 3
    elif angle < -0.1 or pos_horiz > 0.2:
        return 1
    elif vitesse_verti < -0.5:
        return 2
    else:
        return 0
