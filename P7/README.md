# Projet 7: Implémentez un algorithme de scoring

# I Préparation des fichiers pour l'utilisation du répertoire
Pour utiliser les notebooks et les applications inclus dans ce répertoire, les fichiers de données originaux, le modèle ainsi que les échantillons utilisés par les applications doivent être récupérés et/ou générés manuellement. Pour cela, se placer à la racine du répertoire et récupérer les fichiers originaux:

	
	cd <repertoire>
	curl -O https://s3-eu-west-1.amazonaws.com/static.oc-static.com/prod/courses/files/Parcours_data_scientist/Projet+-+Impl%C3%A9menter+un+mod%C3%A8le+de+scoring/Projet+Mise+en+prod+-+home-credit-default-risk.zip
	unzip Projet+Mise+en+prod+-+home-credit-default-risk.zip -d project_files

Vérifier que le fichier config.json est configuré de la manière suivante et le corriger si ce n'est pas le cas:
	
	{ 
		"SAVE_TO":"sample_data",
		"READ_FROM":"project_files",
		"NUM_ROWS":10000
	}	


Attention, le nom du dossier dans lequel les fichiers téléchargés ont été placés et la valeur du champ "READ_FROM" du fichier de config.json doivent coïncider.

Se placer à la racine du répertoire, ouvrir un terminal et commencer par créer un environnement virtuel:

	cd <repertoire>
	pip -m venv venv
	source venv/bin/activate

puis procéder à l'installation des packages requis:

	pip install -r requirements.txt

Lancer le script p7.py:
	
	python p7.py 

Le script va générer automatiquement les fichiers de données nécessaires pour l'entraînement du modèle, entraîne le modèle et calcule les valeurs shap nécessaires à l'interprétation du modèle. Ces fichiers sont placés dans un répertoire nommé sample_data.

Le code est largement inspiré du kernel kaggle disponible à cette adresse: https://www.kaggle.com/c/home-credit-default-risk/data\
L'algorithme de prédiction utilisé est LightGBM https://lightgbm.readthedocs.io/

# II Comment utiliser ce répertoire
## II.1 Contenu du répertoire
Après applications des étapes décrites précédemment, le répertoire doit en principe contenir les fichiers suivants:
- Le notebook contenant l'analyse exploratoire, l'entraînement du modèle et les essais d'interprétation du modèle
- les fichiers lightgbm_with_simple_features.py et p7.py renfermant les scripts nécessaires au fonctionnement du modèle et des applications
- les dossiers prediction_app et dashboard_app contenant les fichiers nécessaires au déploiement des applications en local ou sur le web
- le dossier sample_data contenant les fichiers csv des données, le modèle entraîné et enregistré au format pickle (fitted_lgbm.pkl), et les fichiers contenant les données utiles pour l'interprétation du modèle (shap_values.pkl et base_value.pkl)

## II.2 Procédures pour lancer les applications en local

L'application peut maintenant être lancée:

	cd prediction_app
	python app.py

Ceci lancera l'API de prédiction. Il est nécessaire de lancer l'API de prédiction d'abord car elle est ensuite appelée par l'application dashboard. Pour lancer l'application dashboard, se replacer à la racine, puis:

	cd dashboard_app	
	python home_credit.py

## II.3 Procédures pour déployer les applications sur le web
Les applications peuvent être déployées sur le web via heroku. Les fichiers nécessaires au déploiement (Procfile, requirements.txt, runtime.txt) sont déjà présents et configurés dans les dossiers des applications. Pour effectuer le déploiement, copier le répertoire de l'application à déployer à l'emplacement de votre choix, y placer une copie du dossier sample_data généré précedemment

	cp -r sample_data /chemin/vers/le/repertoire/de/l'application

ou le créer si ce n'est pas le cas:

	cd <repertoire>
	python p7.py
	cp -r sample_data /chemin/vers/le/repertoire/de/l'application

Il faut modifier le fichier config.json ainsi:

	{
		"PATH":"sample_data/"
	}

Pour déployer l'application, suivre les instructions données à cette adresse: https://devcenter.heroku.com/articles/git


## III Prédiction de la capacité de remboursement d'emprunt des clients de Home Credit
Home Credit est une institution financière internationale non bancaire fondée en 1997 en République tchèque et basée aux Pays-Bas. La société opère dans 9 pays et se concentre sur les prêts à tempérament principalement aux personnes ayant peu ou pas d'antécédents de crédit. [Wikipedia](https://en.wikipedia.org/wiki/Home_Credit). A cause de la nature même de sa clientèle, Home Credit a recours à des sources d'informations variées - notamment des informations liées à l'utilisation des services de téléphonie et aux transactions effectuées par les clients - pour prédire la capacité de remboursement des clients. C'est une partie de ces données (anonymisées) que Home Credit a mis en ligne  (https://www.kaggle.com/c/home-credit-default-risk/data), et sur lesquelles le présent travail repose.

### III.1 Présentation du modèle
Les données sont contituées de huit tables, liées les unes aux autres via une ou pusieurs clés comme indiqué sur le shema ci-dessous [home_credit](home_credit.png)
Les descriptions des différentes tables peuvent être consultées [ici](https://www.kaggle.com/c/home-credit-default-risk/data).\
Le modèle utilise toutes ces tables, Les jointures et le _feature engineering_ étant réalisés par la fonction main() du module p7.py, très étroitement inspirée du kernel kaggle [LightGBM with simple features](https://www.kaggle.com/jsaguiar/lightgbm-with-simple-features/script).

L'entraînement du modèle est réalisé en scindant le jeu de données en un jeu de d'entraînement et un jeu de test dans des proportions 0.7 - 0.3, en prenant en compte le déséquilibre des classes grâce au paramètre stratify=True.\
Pour la détermination des meilleurs hyperparamètres, le jeu d'entraînement est lui-même scindé en 5 plis grâce à la méthode StratifiedKold qui permet de prendre en compte le déséquilibre des classes.







