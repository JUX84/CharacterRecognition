# **CharacterRecognition**

Ce logiciel permet d'effectuer de la reconnaisse de caractères dactylographiés grâce à la librairie **OpenCV**.
Il a été développé par *Mehdi MHIRI* et *Bastien DELAVIS* dans le cadre d'un projet pour le Master 2 ILSEN du CERI.

---

## Installation

> **Dépendances:**

> - CMake 3.0 ou mieux
> - OpenCV 3.0.0 ou mieux

Afin d'installer ce projet, effectuez les commandes suivantes:
```
./extract.sh # permet d'extraire les données
cmake . # va vérifier les dépendances puis créer le Makefile
make # compilation du code
```

---

## Utilisation

Le logiciel fonctionne en ligne de commandes et son comportement peut être modifié grâce à des paramètres.
Afin de connaître les différents paramètres disponibles, tapez:
```
./bin/CR
```
Un modèle pré-entrainé est fourni avec le projet, pour l'utiliser et prédire l'image *img/text.png*, tapez:
```
./bin/CR -load full.mdl -predict
```
Il devrait transcrire l'alphabet en se trompant sur la lettre i majuscule qu'il va confondre avec un l minuscule (ce qui, hors contexte, peut arriver à un humain aussi, tout comme la différence entre un zéro et la lettre O, selon les polices).

Ce modèle a été généré grâce à la commande:
```
./bin/CR -process -samples 1016 -save -test
```
Le modèle a été généré en *27 minutes*, en fournissant lors du test un taux de réussite de *78.5%*.

