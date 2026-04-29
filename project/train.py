##
## EPITECH PROJECT, 2026
## G-AIA-401-LYN-4-1-starttrek-7
## File description:
## train
##

import torch
import torch.nn.functional as F

def learn(batch, optimizer, t_net, o_net):
    obs, actions, rewards, next_obs, dones = batch
    with torch.no_grad():
        next_s = t_net(next_obs).max(dim=1).values #meilleur action
        target = rewards + 0.99 * next_s * (1 - dones)    # Bellman : valeur d'une action
    predicted = o_net(obs).gather(1, actions.unsqueeze(1)).squeeze(1) #back to realit"
    loss = F.mse_loss(predicted, target)
    optimizer.zero_grad()
    loss.backward() #relearn
    optimizer.step()
    return loss.item() #sureté