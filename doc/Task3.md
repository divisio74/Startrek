# Task 3 — Ablation

## Démarche

Pour cette task 3 j'ai d'abbord essayer de faire une ablation simple en prenant les `buffer_size` et en fesant chnager le'ordre de grandeur afin d'estimer l'impacte sur l'entrainement.


On a aussi crée un plot.py permettant de visualiser directemnt la différences entres les models au niveau de la buffersize.

## Observation

Ce que j'ai pu observer c'tes que le train avec **90 mille** de buffersize est celui qui a les meilleurs performance alors que je m'attendais a ce que ce sois le **50K**. (voir image ablation)

je m'attendais a ce résualt car 90K est censé etre supereieux aux nombre de transition totral généré d'apres mes calcules donc cela veut dire que le model apprend mieux quand le buffersize contient tout l'historic ce qui n'ets pas vraiment logique.

je vais docn tanté de retrain le model 50K avec différentes seed pour voir si ce n'est pas un probleme de variation.


Malgres les test de 3 seed (0 1 2 ) (voir image ablation2) on constate que les trois reste en dessous de 200 de mean ce qui veut dire que le probleme est ailleurs en attendant nous poursuivrons la suite de cet exercice avec le model a 90K.


