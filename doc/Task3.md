# Task 3 — Ablation

## Démarche

Pour cette task 3 j'ai d'abbord essayer de faire une ablation simple en prenant les `buffer_size` et en fesant chnager le'ordre de grandeur afin d'estimer l'impacte sur l'entrainement.

## Observation

Ce que j'ai pu observer c'tes que le trian avec **90 mille** de buffersize est celui qui a les meilleurs performance alors que je m'attendais a ce que ce sois le **50K**.

## Explication

On peeut expliqyuer ceci car les **90K de buffersize** permette de garder quasiment toutes les transition donc bcp d'ancienne ce qui stabilise et décorelle bcp plus.
