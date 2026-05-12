# Task 2 — AblaAgent (discrete)

# Démarche
Dans un premier temps, j'essaye de comprendre les notions générales du projets et les plus importants. j'implémente une MLP de manière hardcoder, puis je la rends dynamique en créant une fonction qui permet de crée des mlps. Puis ensuite, j'essaye de comprendre les liens entre l'epsilon greedy decay et je l'implémente. J'implémente les deux dans un fichier test.py avant de les mettres dans les fichiers de la branches.
Et c'est là qu'intervient la notion de replay. Au départ, c'était compliqué, mais j'essayee de comprendre ce que c'est en le refaisant. Puis j'essaye de l'implémenter dans le test, pour finir dans la task2.
Une fois toutes les connaissances, acquises, j'essaye de les merges et de les faire fonctionner.

# Observation

Au départ, l'ia n'était pas capable de s'entraîner sur une large zone d'action, puis de les rétrécirs, les résultats étaient palpadres. Mais au fil du temps, j'ai implémenter une méthode qui permet de rétrécir le cham d'action de la machine. Deuxièmement, la machine n'était pas capable de stoquer les infos, ce qui ne l'a rendait pas capable d'apprendre, du coup j'ai crée un tableau de donnée qui permet de stoquer toutes les steps (dans le replay). Ces steps serviront à faire apprendre à notre réseau de neuronnes pour la méthode learn. Elles sont prises au hazard pour permettre à l'ia d'apprendre des règles de déplacements. Learn, est l'étape final du prog de reinforcement learning. Elle reçoit un batch d'informations. Puis calcul la valeur future attendu afin de permettre d'ajuster ses résultats. Elle le compare, puis elle compare l'écart entre le résultat estimé et rendu. Pour finir par se mettre à jour.

# Résultat

Au final, notre petite machine est capable d'apprendre sur un terrain fixé, elle s'améliore lentement, il y a des moments où elle fait qq chose de bien, et ensuite elle fait absolument tout et n'importe quoi. Des moments où elle reste allumé pendant 20s, et truncate le programme. Mais le score augmente au fur et à mesure des séances. Donc elle est capable d'apprendre.