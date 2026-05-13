##
## EPITECH PROJECT, 2026
## G-AIA-401-LYN-4-1-starttrek-7
## File description:
## classification
##


def ending_info(terminated, truncated, obs, last_reward):
    if truncated:
        return "truncated"
    if terminated:
        if (obs[0] > 1 or obs[0] < -1):
            return "out of view"
        if (last_reward >= 50):
            return "Atterrissage Bon !"
        if (last_reward < 0):
            return "crash !"
        return "sleep"
    return "non terminé"

