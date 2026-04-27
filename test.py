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
alunisseur, info = env.reset(seed=42)

if os.path.exists("./videos"):
    shutil.rmtree("./videos")
os.makedirs("./videos")
env = RecordVideo(env, video_folder="./videos", episode_trigger=lambda ep: ep % 10 == 0)

#MACROS
SLEEP = 0
CRASH = 1
OUT_OF_VIEW = 2
TIMEOUT = 3

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

last_reset_step = 0
total_reward = 0
data_: list[episode] = []
for i in range(3000):
    data__: episode

    action = policy(alunisseur)
    alunisseur, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    if is_finished(truncated, terminated) == True:
        lenstep = i - last_reset_step
        data_.append(store_data(total_reward, lenstep, terminated, truncated))
        last_reset_step = i
        print(f"=== Score : {total_reward} === Steps : {i}")

        total_reward = 0
        alunisseur, info = env.reset()

# Fixer les erreurs.
# Faire apprendre à la alunisseur. Ses erreurs. [OK]
#
# Score step
# remettre les trucs que pierre à implémenter pour la step 1 (le yaml, et la manière dont il génère ses episodes, j'aime bcp)


# Pas de YAML config — le sujet exige un fichier YAML par run (env, algo, hypers, seeds).
# Plots manquants

env.close()