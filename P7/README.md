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


# III Prédiction de la capacité de remboursement d'emprunt des clients de Home Credit
Home Credit est une institution financière internationale non bancaire fondée en 1997 en République tchèque et basée aux Pays-Bas. La société opère dans 9 pays et se concentre sur les prêts à tempérament principalement aux personnes ayant peu ou pas d'antécédents de crédit. [Wikipedia](https://en.wikipedia.org/wiki/Home_Credit). A cause de la nature même de sa clientèle, Home Credit a recours à des sources d'informations variées - notamment des informations liées à l'utilisation des services de téléphonie et aux transactions effectuées par les clients - pour prédire la capacité de remboursement des clients. C'est une partie de ces données (anonymisées) que Home Credit a mis en ligne  (https://www.kaggle.com/c/home-credit-default-risk/data), et sur lesquelles le présent travail repose.

## III.1 Présentation du modèle
Les données sont constituées de huit tables, liées les unes aux autres via une ou pusieurs clés comme indiqué sur le shema ci-dessous [home_credit](home_credit.png)
Les descriptions des différentes tables peuvent être consultées [ici](https://www.kaggle.com/c/home-credit-default-risk/data).\
Le modèle utilise toutes ces tables, Les jointures et le _feature engineering_ étant réalisés par la fonction main() du module p7.py, très étroitement inspirée du kernel kaggle [LightGBM with simple features](https://www.kaggle.com/jsaguiar/lightgbm-with-simple-features/script).

Le problème à résoudre concerne **la probabilité qu'un client ne rembourse pas son crédit**. La cible qu'on cherche à prédire est une variable **binaire**: 0 signifie qu'un client rembourse son crédit (no default), 1 que le client ne rembourse pas son crédit (default). Il s'agit donc d'un problème de classification, et on peut relever certaines caractéristiques importantes à prpos du jeu de données:
- un grand nombre de features (plusieurs centaines)
- les features sont de type mixte: continu et catégoriel
- il s'agit d'un jeu de données **déséquilibré**, dans le sens ou la proportion de clients n'ayant pas remboursé leur crédit est d'environ **9%** (et non 50% comme attendu dans le cas d'un jeu de données équilibré). Cela va fortement peser dans les stratégies d'optimisation du modèle.

## III.2 Entraînement du modèle

### III.2.1 Algorithme de prédiction
L'algorithme utilisé est [LightGBM](https://lightgbm.readthedocs.io). Il s'agit d'un algorithme de la famille des arbres de décision boostés et qui possède des caractéristiques le rendant particulièrement efficace, comme la méthode de croissance des arbres ou encore le traitement des variables catégorielles ([voir la doc officielle](https://lightgbm.readthedocs.io/en/latest/Features.html)), en plus des qualités intrinsèques de cette famille d'algorithmes, qui permet, en combinant un grand nombre d'apprenants faibles permet de réaliser un modèle particulièrement performant. 
Les contreparties de cette efficacité sont multiples:
- le fort risque de surapprentissage
- le grand nombre d'hyperparamètres ajustables et dont une détermination optimale nécessite une recherche sur grille dans un espace des paramètres de grande taille. Ainsi, on choisit pour la détermination des meilleurs hyperparamètres l'algorithme RandomSearchCV qui va permettre de parcourir une portion significative de l'espace des paramètres en n'utilisant qu'un nombre restreint d'itérations, permettant ainsi un gain de temps sensible par rapport à la méthode standard GridSearchCV au prix toutefois d'une évaluation plus ou moins suboptimale en comparaison.\

### III.2.2 Optimisation des hyperparamètres
Pour effectuer l'optimisation des hyperparamètres, le jeu de données est scindé une première fois en une paire entraînement/test dans les proportions 0.7/0.3, puis le jeu d'entraînement est lui-même scindé en une autre paire entraînement/validation dans les proportions 0.7/0.3. Ceci est réalisé grâce à la fonction train_test_split de la bibliothèqye scikit-learn en prenant en compte le déséquilibre des classes grâce au paramètre stratify=True, ce qui permet d'avoir des jeux d'entraînement et de test qui reproduisent aussi fidèlement que possible les distributions relatives des deux classes dans le jeu de données global. Chaque modèle est évalué au travers d'une validation croisée sur 5 folds stratifiés (sklearn.model_selection.StratifiedKFolds) afin de prendre en compte le déséquilibre des classes. Les hyperparamètres ajustés sont les suivants:
- le nombre maximal de feuilles (num_leaves) - permet de prévenir le surapprentissage
- le taux d'apprentissage (learning_rate) - permet de trouver un compromis entre efficacité et performance "brute"
- le nombre de boosts (n_estimators) - pour augmenter la peformance
- le nombre de features retenues pour faire croître chaque arbre (colsample_bytree), en fraction du nombre total de features
- le nombre d'individus minimal requis pour faire "pousser" une feuille - prévention du surapprentissage
- le poids affecté aux individus de la classe minoritaire (scale_pos_weight): ce paramètre est pris en compte par la fonction de coût lors de l'entraînement du modèle, en affectant une pénalité plus forte aux éléments de la classe minoritaire incorrectement classifiés
- bagging_fraction, qui consiste à ne construire les arbres que sur un échantillon d'individus tirés au sort (avec remplacement): permet de réduire le risque de surapprentissage

### III.2.3 Fonction de coût et métrique d'évaluation
LightGBM peut-être utilisé pour réaliser des régressions aussi bien que des classifications. Dans le cas qui nous concerne, la fonction de coût est la fonction *log-loss*, ou encore *negative log likelihood* et le *coût* associé à une prédiction s'écrit:

	L(y, p) = -(ylog(p) + (1-y)log(1-p))

où y représente la "vraie" valeur de la cible, et p la probabilité associée à la classe 1. Le coût associé à la totalité du jeu de données est simplement donné par la somme des coûts des échantillons individuels, et les paramètres du modèle sont alors ajustés de manière à minimiser ce nombre. Une fois le modèle entraîné, son évaluation se fait au moyen de la métrique choisie, la plus courante étant la précision (accuracy) dans le cas des problèmes de classification. 

L'une des faiblesse de cette méthode, dans notre cas, vient du caractère déséquilibré du jeu de données: en effet, dans la mesure où la classe minoritaire représente environ 10% des échantillons, on peut obtenir une précision d'environ **90%** simplement en prédisant toujours la classe majoritaire. Fait d'autant plus gênant que la prédiction correcte de la classe minoritaire est cruciale dans notre cas, plus importante que la prédiction correcte de la classe majoritaire - il vaut mieux, pour un organisme de crédit, refuser de prêter à un client qui aurait remboursé (faux positif) que d'accepter de prêter à un client non solvable (faux négatif). Pour éviter ces écueils, on agit sur deux leviers:
- la fonction de coût
- les métriques utilisées pour évaluer la qualité du modèle
 
**la fonction de coût**\
Comme évoqué plus haut, LightGBM possède un paramètre *scale_pos_weight* qui permet de multiplier les echantillons de la classe minoritaire par un poids de manière à pénaliser plus fortement les échantillons de la classe minoritaire incorrectement classifiés. En effet, la fonction de coût s'écrit alors, dans le cas où y=1:
	
	y = 1: L(y, p) = -wlog(p)
 	
où w représente le poids. La prédiction incorrecte est donc associée à un coût w fois plus grand que dans le cas où l'échantillon appartient à la classe majoritaire

	y = 0: L(y, p) = -log(1-p)

Le coût total étant égal à la somme sur tous les échantillons, ceci permet bien d'obtenir une fonction de coût qui pénalisera *in fine* autant les mauvaises classifications associées à chacune des classes si w est ajusté de telle sorte que:
	
	w = nb éch. classe majoritaire/nb éch. classe minoritaire

**la métrique d'évaluation**\
De même, il faut choisir une métrique qui permette de mieux rendre compte de la qualité de classification pour chacune des classes. Plusieurs outils peuvent répondre à cela:
- la matrice de confusion, qui permet de visualiser directement le taux de vrais positifs (sensibilité, ou *recall*), ainsi que le ratio du nombre de vrais positifs sur la somme des vrais positifs et des faux positifs (précision). Ces deux nombres peuvent être combinés pour former le *F-score*. Nous utilisone le *F-score* pour évaluer la qualité du modèle.
- Area Under Receiver Operating Characteristic (AUROC): la ROC est une courbe construite en reportant, pour chaque valeur de seuil de discrimination, le taux de vrais positifs (fraction des positifs qui sont effectivement détectés, précision) en fonction du taux de faux positifs (fraction des négatifs qui sont incorrectement détectés, antispécificité). Cette courbe part du point (0, 0) où tout est classifié comme "négatif", et arrive au point (1, 1) où tout est classifié "positif". Un classifieur aléatoire tracera une droite allant de (0, 0) à (1, 1), donnant une AUROC égale à 0.5, et tout classifieur "meilleur" sera représenté par une courbe située au-dessus de cette droite et possèdera ainsi une AUROC supérieure à 0.5. Nous utilisons également cette métrique pour évaluer le modèle.

### III.2.4 Préparation des jeux de données pour optimiser la qualité du modèle
Une autre stratégie peut être employée pour corriger le déséquilibre des classes et ainsi permettre de meilleurs résultats: il s'agit de modifier le jeu de données en sur-échantillonant la classe minoritaire - via la création de nouveaux individus -, et/ou de sous échantillonner la classe majoritaire en effectuant un tirage aléatoire.
Cette stratégie est mise en oeuvre à l'aide la bibliothèque imblearn. Pour le sur-échantillonnage, SMOTENC est choisi pour sa capacité à prendre en compte les variables catégorielles. 

### III.2.5 Résumé de la méthodologie d'entraînement
En réssumé, les étapes mises en oeuvre sont les suivantes: 
1. Chargement d'un extrait du jeu de données. 
2. Séparation entraînement/test une fois pour la recherche sur grille des meilleurs paramètres, puis sélection des features les plus pertinentes par la méthode des permutations. Cette sélection doit se faire sur un modèle aussi bon que possible, ce qui rend nécessaire la recheche sur grille de bons paramètres déjà à cette étape. Cette étape permet d'obtenir une réduction de 500+ à environ 20 features.
3. Chargement du jeu de données complet, avec restriction aux features sélectionnées auparavant.
4. Oversampling avec SMOTENC et undersampling (imblearn). 
5. Deuxième recherche sur grille, puis entraînement sur le jeu d'entraînement complet avec les meilleurs paramètres.

# IV Interprétation du modèle
On cherche, dans un but de compréhension du modèle mais aussi de communication, à interpréter le modèle c'est à dire à déterminer:
* les variables les plus importantes dans notre jeu de données, c'est-à-dire celles qui ont le plus d'impact sur le score global obtenu par l'estimateur - ici F1 et/ou AUROC. Il s'agit alors de l'interprétation globale.
* Pour chaque prédiction, quelles sont les variables qui ont principalement conduit l'estimateur à faire la prédiction en question. Il s'agit alors de l'interprétation locale.

LightGBM représente une sorte de "boîte noire": il s'agit d'un algorithme *non-paramétrique*, et contrairement au cas de la régréssion linéaire nous n'avons pas à notre disposition des coefficients (évalués lors de l'entraînement) associés à chaque variable et qui permettent directement d'interpréter le poids affecté à chaque variable par le modèle. On choisit donc, pour l'interprétation globale, d'utiliser le calcul des *feature importances* via la méthode des permutations. En effet, cette méthode est agnostique - elle ne repose pas sur un modèle en particulier - et n'est pas sujette à certains biais qui peuvent être rencontrés lors du calcul des *feature importances* via par exemple l'évaluation du coefficient de Gini - méthode standard dans le cas des algorithmes à base d'arbres de décision. La méthode des permutations consiste à permuter, aléatoirement, les valeurs d'une variable et à observer la diminution de score qui en résulte. Plus la diminution est importante, plus la variable est importante.
Dans notre cas, cette méthode est utilisée deux fois: une première fois pour sélectionner les variables les plus  imoportantes, une deuxième fois pour réaliser l'interprétation globale.

Pour l'interprétation locale du mmodèle, on fait appel à la bibliothèque SHAP, et au calcul des *shap values*. Il s'agit ici d'une méthode directement adaptée de la théorie des jeux, et qui consiste à évaluer l'influence de chaque variable sur la prédiction faite par le modèle (ici la probabilité de non-remboursement). Le point de départ de cette méthode est l'évaluation d'un score de base, qui correspond en fait à l'espérance de la variable cible, et qui peut s'interpréter comme la probabilité qu'un individu tiré au hasard ne rembourse pas le crédit. L'estimateur, parce qu'il a été entraîné, renvoie une prédiction sensiblement différente et le calcul des *shap values* permet d'associer à chaque variable sa part de la différence entre la probabilité calculée par l'estimateur et l'espérance. Les détails de l'implémentation de cette bibliothèque peuvent être consultés [ici](https://shap-lrjball.readthedocs.io/en/latest/).

# V Limites et améliorations possibles
Même si l'estimateur entraîné réalise une meilleure classification que le classifieur aléatoire (AUROC de 0.76 vs. 0.5 pour le classifieur aléatoire), le modèle souffre quand même d'une performance limitée, comme l'indique le score F1 de 0.3 obtenu sur le jeu de test. On peut imaginer les pistes suivantes pour améliorer les résultats: 

* Entraînement de l'estimateur:
	- la capacité du modèle à gérer directement les variables catégorielles n'est pas exploitée, mais il serait intéressant de voir si cela conduit à de meilleures 	performances
	- bien qu'une recherche sur grille ait été effectuée, il se peut que les valeurs retenues correspondent à un extremum local. Une recherche plus exhaustive pourrait peut-être conduire à de meilleurs résultats
	- Déterminer de nouvelles variables via des considérations métiers
	- Modifier le seuil de discrimination pour augmenter le taux de vrais positifs, tout en maintenant aussi bas que possible le taux de faux négatifs.

Le dashboard présente un certain nombre d'informations, mais peut sans doute être amélioré pour rendre les résultats encore plus lisibles, par exemple en rendant possible la présentation de graphiques autres que les boîtes à moustache ou les nuages de points.








