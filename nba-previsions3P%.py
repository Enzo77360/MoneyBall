from sklearn.metrics import mean_squared_error
import xgboost
import pandas as pd
import numpy as np

# Chargement du dataset depuis le fichier CSV Entrainement sur annee 2018
dataset = pd.read_csv("C:\\Users\\enzos\\PycharmProjects\\MoneyBall\\data-nba-player-2018.csv")

# Supprimer les colonnes non numériques et non pertinentes
dataset = dataset.drop(columns=["Player", "Pos", "Tm", "Player-additional"])

# Remplacer les valeurs manquantes par 1 dans la colonne '3P%'
dataset['3P%'] = dataset['3P%'].fillna(1)

# Séparation des variables indépendantes (X) et dépendante (y)
x = dataset.drop(columns=["3P%"])
y = dataset["3P%"]

# Initialisation du modèle XGBoost Regressor
model = xgboost.XGBRegressor(objective='reg:squarederror')

# Entraînement du modèle sur les données
model.fit(x, y)

# Anticipation sur annee 2019
test_dataset = pd.read_csv("C:\\Users\\enzos\\PycharmProjects\\MoneyBall\\stats-nba-player-2019.csv")

# Supprimer les colonnes non numériques et non pertinentes
test_dataset = test_dataset.drop(columns=["Player", "Pos", "Tm", "Player-additional"])

# Remplacer les valeurs manquantes par 1 dans la colonne '3P%'
test_dataset['3P%'] = test_dataset['3P%'].fillna(1)

x_test = test_dataset.drop(columns=["3P%"])
y_test = test_dataset["3P%"]

pred = model.predict(x_test)

# Ajout des prédictions au DataFrame de test
test_dataset['Predicted 3P%'] = pred

# Seuil pour le pourcentage à trois points (à adapter selon vos critères)
seuil_3p = 0.35

# Filtrer les joueurs avec un pourcentage à trois points supérieur au seuil
meilleurs_shooteurs = test_dataset[test_dataset['Predicted 3P%'] > seuil_3p]

# Trier les meilleurs shooteurs par ordre croissant des minutes jouées
meilleurs_shooteurs_tries = meilleurs_shooteurs.sort_values(by='MP', ascending=True)

# Réinitialiser l'index de ligne pour avoir un identifiant unique pour chaque joueur
meilleurs_shooteurs_tries.reset_index(drop=True, inplace=True)

# Imprimer le nom des joueurs et leur 3P% dans l'ordre croissant des minutes jouées
print("Liste des meilleurs shooteurs à trois points (3P%) par ordre croissant des minutes jouées :")
for index, row in meilleurs_shooteurs_tries.iterrows():
    print(f"Joueur {index+1} - 3P% : {row['Predicted 3P%']}, Minutes jouées : {row['MP']}")
