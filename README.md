*Réaliser par Mathéo GUILBERT*
*-- Octobre 2025 --*

# Journal de bord

## Introduction

Le jeu de données utilisé est `PleIAs/French-PD-Books`.

## Partie 1 : Classification

**Deux approches :**

1. Entrainement du modèle sur des données brutes vectorisées
2. Entrainement sur des données avec analyse de patrons sémantiques et stylistique préalable

**Pronostiques :**

Je pense que l'approche 2 donnera de meilleurs résultat que la première. Cepandant la deuxième nécessitera un travail d'exploration des données plus important.

Je pense également que la première approche peut facilement poser un problème de surapprentissage sur les indicateurs temporelles explicites tel que les dates ou les nom d'époque (renaissance, moyen age, ...).

Pour la méthode de vectorisation j'essaie deux méthodes :

1. TF-IDF
2. Embeddings

Pour ce qui est des modèles utilisé :

TF-IDF --> c'est le classifier
Embeddings -->  dangvantuan/sentence-camembert-base <-- Pour mon PC
                dangvantuan/sentence-camembert-large <-- Pour Morgoth

## Partie 2 : Génération