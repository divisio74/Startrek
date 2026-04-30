=============================================================
========================= Task 2 ============================
=============================================================

__Agent (discrete)__
1. Crée un MLP (Multi-layer perception) (La machine)
2. Implémenter une mémoire (Replay)
3. Donner la capacité d'apprendre (learn)

__Dépendance à installer__ :
1) import torch.nn

# C'est quoi une MLP ?
On définit l'identité de l'agent. Le réseau neuronal qui est composé d'une couche d'entrée, de sortie, et des hiddens layers.

Les hiddens layers servent à faire le choix de la machine.

définir :
Big_brain = Sequential (        -> Regroupe plusieurs layers.
    Layer (entrée)
    Relu
    Layer (hidden)
    Relu
    Layer (sortie)
)

Autrement dit,
Big_brain = nn.Sequential (
    nn.Linear(8, 128) (pour 8 inpouts (obs))
    nn.ReLu()
    nn.Linear(128, 128)
    nn.ReLu()
    nn.Linear(123, 4) (pour 4 sorties (4 actions))
    nn.ReLu()
)

# C'est quoi un replay ?
Un replay, c'est une liste d'expérience. (obs, actions, rewards, next_obs, dones)

buffer = deque(maxlen=100)           #liste chaîné de max 100
buffer.add(experience)
buffer.len
buffer.show                          #show les datas

C'est l'idée principale.
En pratique :

class replay():
    def __init__(self):
        self.buffer = deque(maxlen=100)
    def push(self, experience):
        self.buffer.append(exprience)
    def show():
        baffa = random.sample(self.buffer, 128)     # prendre step pif d'ep au pif -> rule gen
        obs, actions, rewards, next_obs, dones = zip(baffa)             # tab de mm val
        return (
            obs
            actions
            rewards
            next_obs
            dones
        )
        return baffa
    def len():
        return slef.buffer.__len__


# MOdification fix pierre:

- Fix arg in run_episdode for separate train and eval properly.

- Add logg in csv way.

- Add a new params config in yaml : model_path: models/epsilon_seed0.pt
