##
## EPITECH PROJECT, 2026
## G-AIA-401-LYN-4-1-starttrek-7
## File description:
## replay
##

import random
from collections import deque

import torch

BATCH_SIZE = 128
BUFFER_SIZE = 50000


class Replay:
    def __init__(self):
        self.buffer = deque(maxlen=BUFFER_SIZE)

    def push(self, obs, action, reward, new_obs, done):
        self.buffer.append((obs, action, reward, new_obs, done))

    def data_for_train(self):
        batch = random.sample(self.buffer, BATCH_SIZE)
        obs, actions, rewards, next_obs, dones = zip(*batch)
        return (
            torch.tensor(obs, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(next_obs, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32),
        )

    def ready(self):
        return len(self.buffer) >= BATCH_SIZE

    def __len__(self):
        return len(self.buffer)
