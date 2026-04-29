##
## EPITECH PROJECT, 2026
## RL
## File description:
## main
##

import gymnasium as gym
from gymnasium.wrappers import RecordVideo #Pour la vidéo
from dataclasses import dataclass #Pour les dataclasses
import shutil, os # Juste pour effacer et remettre la vidéo
import torch.nn as nn # MLP
from collections import deque # liste chaîné autogestionné
import random
import torch
import torch.nn.functional as F

# env = gym.make("LunarLander-v3", render_mode ="rgb_array")
# alunisseur, info = env.reset(seed=42)

# if os.path.exists("./videos"):
#     shutil.rmtree("./videos")
# os.makedirs("./videos")
# env = RecordVideo(env, video_folder="./videos", episode_trigger=lambda ep: ep % 10 == 0)

#MACROS
SLEEP = 0
CRASH = 1
OUT_OF_VIEW = 2
TIMEOUT = 3
state_dim = 8
n_actions = 4
nn_size = 128
BATCH_SIZE = 128
# epsilon = 0.1

#DATASTRUCT
@dataclass
class episode:
    len_: int = 0
    end_: int = 0
    sco_: float = 0

def policy(obs):
    x, y, vx, vy, theta, theta_dot, left_leg_contact, right_leg_contact = obs
    if right_leg_contact == 1:
        return 1
    elif left_leg_contact == 1:
        return 1
    elif vx < 0.01:
        return 1
    elif vy < 0.01:
        return 2
    elif x < 0.2:
        return 1
    elif y < 0.2:
        return 1
    return 0

def is_finished(trun, term):
    if trun == True or term == True:
        return True
    return False

def store_data(sco, len, term, trun):
    e = episode()
    e.len_ = len
    e.sco_ = sco
    if (term == True and sco < -50):
        e.end_ = CRASH
    elif (term == True and sco < 200 and sco > 0):
        e.end_ = OUT_OF_VIEW
    elif (term == False and trun == True):
        e.end_ = TIMEOUT
    else:
        e.end_ = SLEEP
    return e

o_net = nn.Sequential(
    nn.Linear(state_dim, nn_size), nn.ReLU(),
    nn.Linear(nn_size, nn_size), nn.ReLU(),
    nn.Linear(nn_size, n_actions)
)

t_net = nn.Sequential(
    nn.Linear(state_dim, nn_size), nn.ReLU(),
    nn.Linear(nn_size, nn_size), nn.ReLU(),
    nn.Linear(nn_size, n_actions)
)

class Replay:
    def __init__(self):
        self.buffer = deque(maxlen=50000)
    def push(self, obs, action, reward, new_obs, done):
        self.buffer.append((obs, action, reward, new_obs, done))
    def data_for_train(self):
        random_xp_tuple = random.sample(self.buffer, BATCH_SIZE)
        obs, actions, rewards, next_obs, dones = zip(*random_xp_tuple)
        return (
            torch.tensor(obs, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(next_obs, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32)
        )
    def ready(self):
        return self.buffer.__len__() >= BATCH_SIZE
    def __len__(self):
        return self.buffer.__len__()

def ai_nn(alunisseur, epsilon, env):
    if random.random() < epsilon:
        return env.action_space.sample()  # exploration
    else:
        obs_tensor = torch.tensor(alunisseur, dtype=torch.float32)
        return o_net(obs_tensor).argmax().item()


optimizer = torch.optim.Adam(o_net.parameters(), lr=1e-3)
replay = Replay()
t_net.load_state_dict(o_net.state_dict())  # copie initiale

def learn(batch):
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

def run():
    list_ep: list[episode] = []
    epsilon = 1.0  # commencer à 1.0, pas 0.1
    step = 0       # compteur global de steps

    env = gym.make("LunarLander-v3", render_mode ="rgb_array")

    if os.path.exists("./videos"):
        shutil.rmtree("./videos")
    os.makedirs("./videos")
    env = RecordVideo(env, video_folder="./videos", episode_trigger=lambda ep: ep % 10 == 0)
    alunisseur, info = env.reset(seed=42)
    for ep in range(40):
        total_reward = 0
        ep_step = 0
        alunisseur, info = env.reset(seed=42)

        while True:
            obs_avant = alunisseur
            action = ai_nn(alunisseur, epsilon, env)
            alunisseur, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            ep_step += 1
            step += 1

            replay.push(obs_avant, action, reward, alunisseur, terminated)

            if replay.ready():
                batch = replay.data_for_train()
                learn(batch)

            if step % 100 == 0:
                t_net.load_state_dict(o_net.state_dict())

            if terminated or truncated:
                list_ep.append(store_data(total_reward, ep_step, terminated, truncated))
                epsilon = max(epsilon * 0.995, 0.01)
                break
    env.close()
    return list_ep

if __name__ == "__main__":
    results = run()
    for i, ep in enumerate(results):
        print(f"Episode {i:3d} | score: {ep.sco_:8.2f} | steps: {ep.len_:4d} | end: {ep.end_}")


# Replay.push(alunisseur, reward, alunisseur, "caca")