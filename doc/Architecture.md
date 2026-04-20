# Architecture

==============================================================================================================

## Tasks
__Baseline__
1. Crée une action random de la machine
2. Faire un simple programme ("heuristique") qui va *essayer* de gérer l'atterrissage
3. Sauvegarder les données qq part (score, cmb de step ça dure, pk ça a ended (termination / truncation))
4. Crée des vidéos avec "gymnasium[other]"

__Dépendance à installer__ :
python3 -m venv venv.
source venv./bin/activate
pip install swig
pip install gymnasium
pip install "gymnasium[box2d]"

__Agent (discrete)__
1. 

==============================================================================================================

# Reinforcement learning
__C'est quoi ?__ :
Apprentissage de l'ia dans son environnement.

__C'est composé de quoi ?__:
- Environnement
- Score
- L'IA (ou Agent)

__Quel processus ?__:
Agent --> Step --> Environnement réagit : Environnement donne un score et un nouvel état --> Agent

==============================================================================================================

# Step (env.step())
1) __Machine_state / observation__
Valeurs de l'atterisseur
[ 0.013,   # x (coordonnées : 0 centre de l'axe x)
  1.412,   # y (coordonnées : 0 au sol)
 -0.231,   # vx (vitesse vers la gauche droite (axe x))
 -0.187,   # vy (vitesse vers le haut bas (axe y))
  0.004,   # angle (du véhicule)
 -0.012,   # vitesse angulaire (vitesse )
  0.0,     # patte gauche (bool, 1 = posé au sol)
  0.0 ]    # patte droite (bool, 1 = posé au sol)

2) __Score / reward__
Note donné par l'environnement à l'agent en fonction de:
- 1. Zone d'atterrissage (plus c'est proche, plus y'a de points)
- 2. Vitesse (Expected Slowest possible)
- 3. Inclinaison (Expected 0)
- 4. Moteurs (Expected least time possible)
- 5. Contact des pattes (1 pattes = 10 points)
- 6. Atterissage (+100 points) / Crash (-100  points)

3) __Terminated (type: bool)__
Mauvaise manoeuvre (Machine sors du cadre / Les deux pieds touchent pas le sol)
- si Machine sors du cadre :
    - vrai : Terminated = true
    - false : Terminated = false
- si les deux pattes ne touchent pas le sol (le corp du coup):
    - vrai : Terminated = true
    - false : Terminated = false

4) __Truncated (type: bool)__
Autre mauvaise manoeuvre (fin forcée par le jeu)
example principal : si l'agent prends trop de temps
- Si step > 500 (max imposé):
    - vrai: Truncated = true
    - false : Truncated = false

+------------+-----------+--------------------------------------+
| terminated | truncated | Situation                            |
+------------+-----------+--------------------------------------+
| False      | False     | Episode en cours, rien de special    |
| True       | False     | Fin naturelle (crash ou atterrissage)|
| False      | True      | Timeout (1000 steps depasses)        |
| True       | True      | Impossible en pratique               |
+------------+-----------+--------------------------------------+