# Prévisions des Meilleurs Shooteurs à Trois Points en NBA

Ce script Python utilise le machine learning pour prédire les meilleurs shooteurs à trois points en NBA. Les données utilisées proviennent de la saison NBA 2018 pour l'entraînement et de la saison NBA 2019 pour les prédictions.

## Données

Les données utilisées dans ce projet sont extraites du site [basketball-reference.com](https://www.basketball-reference.com/leagues/NBA_2019_totals.html), qui est une source fiable pour les statistiques et les performances des joueurs de la NBA.

- **Données d'entraînement** : Les statistiques des joueurs de la saison NBA 2018 sont stockées dans le fichier CSV "data-nba-player-2018.csv". Ces données comprennent des informations telles que les points marqués, les rebonds, les passes décisives, et le pourcentage de réussite à trois points.
  
- **Données de test** : Les performances des joueurs de la saison NBA 2019 sont disponibles dans le fichier CSV "stats-nba-player-2019.csv". Ces données servent à évaluer la capacité du modèle à prédire les meilleurs shooteurs à trois points pour cette saison.

## Prétraitement des Données

Avant d'entraîner le modèle et de faire des prédictions, les données sont prétraitées de la manière suivante :

- **Suppression des Colonnes Non Pertinentes** : Les colonnes telles que le nom du joueur, la position, l'équipe, etc., qui ne sont pas utilisées pour les prévisions, sont supprimées des ensembles de données.
  
- **Remplacement des Valeurs Manquantes** : Les valeurs manquantes dans la colonne du pourcentage à trois points ('3P%') sont remplacées par la valeur 1. Cela est fait pour éviter tout impact négatif sur l'entraînement du modèle.

## Modèle Utilisé

Le modèle utilisé pour ce projet est XGBoost, une bibliothèque de machine learning extrêmement performante pour les problèmes de régression. XGBoost est choisi pour sa capacité à gérer efficacement de grandes quantités de données et à produire des prédictions précises.

## Exécution du Script

Pour exécuter le script, suivez ces étapes :

1. Assurez-vous d'avoir Python installé sur votre système.
2. Installez les bibliothèques nécessaires en utilisant la commande suivante :
   ```
   pip install pandas numpy xgboost
   ```
3. Exécutez le script `nba-previsions3P%.py` en utilisant Python.

Le script générera une liste des meilleurs shooteurs à trois points (3P%) dans l'ordre décroissant des minutes jouées, en affichant leurs noms, leur pourcentage à trois points prédit et leurs minutes jouées.
