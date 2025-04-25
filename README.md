# NOVA : Numerical Optimization & Validation Assistant

NOVA est une application conçue pour les Data Scientists qui en ont marre de perdre du temps sur le nettoyage de données et les tests de modèles. Elle offre une interface simple pour :

- importer une ou deux bases structurées,

- configurer les étapes de prétraitement,

- visualiser les étapes du processus

- tester plusieurs modèles prédictifs supervisés,

- optimiser automatiquement les hyperparamètres,

- comparer les performances de manière rigoureuse.

Avec NOVA, vous vous concentrez sur l’essentiel : prendre des décisions basées sur des résultats solides.

## Prérequis

Pour pouvoir exécuter les script sur votre machine personnelle :
**Installation de Python** : Veuillez installer Python dans sa version 3.12.8 Vous pouvez la télécharger sur [python.org](https://www.python.org/).
   
## Structure du dépôt 

- __src__ :  
    - **`\app`**
        - **`\_main_.py`** : Script utilisé pour le fonctionnement de l'application
    - **`\tools`**
        - **`\local.py`** : Script utltisé comme base et tests locaux pour le développement de l'application. Utilisable sur machine local pour faire des ajustements et les tester.
- __README.md__ : Le message qui décrit le projet         
- __requirements.txt__ : Liste des modules nécessaires à l'exécution des codes Python.      

## Installation

1. **Clonez le dépôt GitHub sur votre machine locale:** 
```bash
git clone https://github.com/Alfex-1/NOVA.git
```

2. **Installez les dépendances requises:**

Pour Python, insérez cette ligne de commande dans le terminal :
```bash
pip install -r requirements.txt
```