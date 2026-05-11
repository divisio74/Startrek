# Task 3 — 95%CI check

## Démarche

Pour cette task 4 nous voullons obtenir un certain de taux de fiabilité considéré comme reproductible : c'ets a dire qu'il faut au minimum 5 config Avec différentes seed et evaluer si notre policy conserve de bonne perf sur les autre seed et si il n'y a pas de trop grosse variation de résultats entre les seeds.

Nous allons donc amiliorer notre plot.py afin qu'il puisse comparé les différentes config que nous allons vouloir comparé en prenant comme hyperparams différentiateur la seed .
pour les config je prend le model de 90K de buffer size car c'est celui qui a été le plus perofrmant durant la tache 3.

## Observation

On observe donc avec plot2.py qui est le scripte python nous permettant d'afficher les résultat des analyse de la variation de notre model avec différentes seed :

`python project/plot2.py logs/eval_*_test_90K_*.csv`

![alt text](image.png)

on consate bien que notre model est totalement reproductible a 95% sur les 5 premieres seed !

On peut dans un deuxiéme temps aussi bien verifier que sur chaque seed on obtiens bien un nombre superieur A 200 de mean return .


## Pour résumer :

la reproductibilité de notre model c'est le fait que les performance ne doivent pas etre vraiment impacté par la seed : en math ca donne -> on calcule la moyenne et apres on calcule les ecarts type ,si l'ecart moyen de chaque seed  avec la moyenne est supériuer 5% du total alors ce n'ets pâs reproductible .

