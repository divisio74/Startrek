# Task 3 — Ablation

## Démarche

Pour cette task 3 j'ai d'abbord essayer de faire une ablation simple en prenant les `buffer_size` et en fesant chnager le'ordre de grandeur afin d'estimer l'impacte sur l'entrainement.


On a aussi crée un plot.py permettant de visualiser directemnt la différences entres les models au niveau de la buffersize.

## Observation

Ce que j'ai pu observer c'tes que le train avec **90 mille** de buffersize est celui qui a les meilleurs performance alors que je m'attendais a ce que ce sois le **50K**. (voir image ablation)

je m'attendais a ce résualt car 90K est censé etre supereieux aux nombre de transition totral généré d'apres mes calcules donc cela veut dire que le model apprend mieux quand le buffersize contient tout l'historic ce qui n'ets pas vraiment logique.

je vais docn tanté de retrain le model 50K avec différentes seed pour voir si ce n'est pas un probleme de variation.


Malgres les test de 3 seed (0 1 2 ) (voir image ablation2) on constate que les trois reste en dessous de 200 de mean ce qui veut dire que le probleme est ailleurs en attendant nous poursuivrons la suite de cet exercice avec le model a 90K.

PS: une possibilité a envisager est que le nombre d'epoch n'ets peut etre pas suffisant pour le 50 mais suffisant pour le 90K car on voit sur les premiers plot que j'avais fait j'avais analyser les mean return du train et j'avais vue que les models 50 et 90 convergeait a la fin pour faire le break avec les autres mais que peut etre le 50K n'a pas eu le temps de se stabiliser alors que celui de 90K oui ... c'tes une theorie .


