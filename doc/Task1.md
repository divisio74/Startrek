# Rapport — StartTrek (LunarLander-v3) TASK 1

> **Note** : à rendre au format PDF.

---

## Rappel des rewards de l'environnement Gym

| Évènement | Reward |
|---|---|
| Atterrissage réussi | **+100** |
| Crash | **-100** |
| Jambe au sol (par jambe) | **+10** |
| Moteur principal allumé (par step) | **-0.3** |
| Moteur latéral allumé (par step) | **-0.03** |

Le coût en carburant pénalise implicitement les épisodes longs.

---

## Jour 1 — Task 1 : politique heuristique

### Approche

Pour la première fonction heuristique, je suis parti d'une base simple en n'utilisant que deux observations :

- `obs[4]` → l'**angle** de l'engin par rapport au sol
- `obs[3]` → la **vitesse verticale** de l'engin

### Premier test

Avec ces deux signaux et le `render_mode = "human"`, le résultat n'était pas horrible mais souffrait d'un problème sur l'**axe horizontal** : l'engin dérivait latéralement.

### Correction

J'ai eu l'idée d'activer les **moteurs latéraux** lorsque l'engin dépasse certaines bornes horizontales. Premier essai : j'avais confondu le moteur droit et le moteur gauche, donc l'engin partait dans le mauvais sens.

Après inversion, le module reste **pile dans l'axe** et continue de corriger l'angle correctement.

---

