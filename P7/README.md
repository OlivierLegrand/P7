# Note méthodologique
# Projet 7: Implémentez un algorithme de scoring

Dans le cas où ce répertoire a été téléchargé depuis github, les fichiers de données doivent être récupérés manuellement.
Le code est largement inspiré du kernel kaggle disponible à cette adresse: https://www.kaggle.com/c/home-credit-default-risk/data\
L'algorithme de prédiction utilisé est LightGBM https://lightgbm.readthedocs.io/\

## I Méthodologie d'entraînement du modèle (2 pages maximum)
Le modèle utilisé ici requiert _a minima_ l'utilisation de la table application_train.csv, mais il fonctionne aussi bien si les autres tables sont jointes à la première.\
Les jointures sont réalisées par la fonction application_train_test du module lightgbm_with_simple_features.\

L'entraînement du modèle est réalisé en scindant le jeu de données en un jeu de d'entraînement et un jeu de test dans des proportions 0.7 - 0.3, en prenant en compte le déséquilibre des classes grâce au paramètre stratify=True.\
Pour la détermination des meilleurs hyperparamètres, le jeu d'entraînement est lui-même scindé en 5 plis grâce à la méthode StratifiedKold qui permet de prendre en compte le déséquilibre des classes.





