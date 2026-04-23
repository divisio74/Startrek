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

env = gym.make("LunarLander-v3", render_mode ="rgb_array")
Machine_state, info = env.reset(seed=42)

if os.path.exists("./videos"):
    shutil.rmtree("./videos")
os.makedirs("./videos")
env = RecordVideo(env, video_folder="./videos", episode_trigger=lambda ep: ep % 10 == 0)

#MACROS
NOT_TERMINATED = 0
TERMINATED = 1
NOT_TRUNCATED = 2
TRUNCATED = 3

#DATASTRUCT
@dataclass
class episode:
    len_: int = 0
    term_: int = 0
    sco_: float = 0

def protoype_rule(obs):
    x, y, vx, vy, theta, theta_dot, left_leg_contact, right_leg_contact = obs
    if right_leg_contact == 1:
        return 1
    elif left_leg_contact == 1:
        return 1
    elif vx < 0.01:
        return 1
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
    if (term == True and trun == False):
        e.term_ = TERMINATED
    else:
        e.term_ = NOT_TERMINATED
    if (term == False and trun == True):
        e.trun_ = TRUNCATED
    else:
        e.trun_ = NOT_TRUNCATED
    return e

last_reset_step = 0
total_score = 0
for i in range(1000):
    data_: list[episode] = []
    data__: episode

    action = env.action_space.sample()
    Machine_state, score_step, terminated, truncated, info = env.step(action)
    total_score = score_step
    if is_finished(truncated, terminated) == True:
        lenstep = i - last_reset_step
        data_.append(store_data(score_step, lenstep, terminated, truncated))
        last_reset = i
        print(f"=== Score : {total_score} === Steps : {i}")
        Machine_state, info = env.reset()


env.close()