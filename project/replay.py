
import argparse
import yaml
import gymnasium as gym
from handle_ep import run_episode
from policies import first_policy
from collections import Counter
import torch.nn as nn
from collections import deque # liste chaîné autogestionné
import torch
import random

NB_HIDDEN = 128

class Replay:
    def __init__(self):
        self.buffer = deque(maxlen=50000)
    def push(self, obs, action, reward, new_obs, done):
        self.buffer.append((obs, action, reward, new_obs, done))
    def data_for_train(self):
        random_xp_tuple = random.sample(self.buffer, NB_HIDDEN)
        obs, actions, rewards, next_obs, dones = zip(*random_xp_tuple)
        return (
            torch.tensor(obs, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(next_obs, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32)
        )
    def ready(self):
        return self.buffer.__len__() >= NB_HIDDEN
    def __len__(self):
        return self.buffer.__len__()
