# Note méthodologique
# Projet 7: Implémentez un algorithme de scoring

Dans le cas où ce répertoire a été téléchargé depuis github, les fichiers de données doivent être récupérés manuellement.
Le code est largement inspiré du kernel kaggle disponible à cette adresse: https://www.kaggle.com/c/home-credit-default-risk/data\
L'algorithme de prédiction utilisé est LightGBM https://lightgbm.readthedocs.io/

# Comment utiliser ce répertoire
## I Contenu du répertoire
Ce répertoire contient les fichiers suivants:
- Le notebook contenant l'analyse exploratoire, l'entraînement du modèle et les essais d'interprétation du modèle
- les fichiers lightgbm_with_simple_features.py et p7.py renfermant les scripts nécessaires au fonctionnement du modèle et des applications
- les dossiers prediction_app et dashboard_app contenant les fichiers nécessaires au déploiement des applications en local ou sur le web
- le dossier sample_data contenant les fichiers csv des données, le modèle entraîné et enregistré au format pickle (fitted_lgbm.pkl), et les fichiers contenant les données utiles pour l'interprétation du modèle (shap_values.pkl et base_value.pkl)

## II Procédures pour lancer les applications en local
Après avoir cloné le répertoire dans le répertoire de votre choix, ouvrir un terminal et entrer les commandes suivantes:
- créer un environnement virtuel à l'aide de la commande pip -m venv venv
- pip install -r requirements.txt
puis:
- cd prediction_app
- python app.py
Ceci lancera l'API de prédiction. Il est nécessaire de lancer l'API de prédiction d'abord car elle est ensuite appelée par l'application dashboard. Pour lancer l'application dashboard, se replacer à la racine, puis:
- cd dashboard_app
- python home_credit.py

## III Procédures pour déployer les applications sur le web
Les applications peuvent être déployées sur le web via heroku. Les fichiers nécessaires au déploiement (Procfile, requirements.txt, runtime.txt) sont déjà présents et configurés dans les dossiers des applications. Pour effectuer le déploiement:
- copier le(s) répertoire(s) de(s) (l')application(s) à déployer à l'emplacement de votre choix
- Suivre les instructions données à cette adresse: https://devcenter.heroku.com/articles/git

## I Méthodologie d'entraînement du modèle (2 pages maximum)
Le modèle utilisé ici requiert _a minima_ l'utilisation de la table application_train.csv, mais il fonctionne aussi bien si les autres tables sont jointes à la première.\
Les jointures sont réalisées par la fonction application_train_test du module lightgbm_with_simple_features.

L'entraînement du modèle est réalisé en scindant le jeu de données en un jeu de d'entraînement et un jeu de test dans des proportions 0.7 - 0.3, en prenant en compte le déséquilibre des classes grâce au paramètre stratify=True.\
Pour la détermination des meilleurs hyperparamètres, le jeu d'entraînement est lui-même scindé en 5 plis grâce à la méthode StratifiedKold qui permet de prendre en compte le déséquilibre des classes.







