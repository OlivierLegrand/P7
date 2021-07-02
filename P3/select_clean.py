import pandas as pd
import weight_to_energy, clean

def select_clean(dataset="./en.openfoodfacts.org.products.csv"):
  '''Fonction qui prend en entrée un jeu de donées dataset au format csv, et renvoie un dataframe ne contenant
  que les variables sélectionnées et sans valeurs aberrantes'''
    
    # 1. Chargement du jeu de données
    df = pd.read_csv("./en.openfoodfacts.org.products.csv")
    
    # 2. Sélection des variables
    df1 = df.copy()
    df1.dropna(subset="product_name", inplace=True)
    selected_cols = ['product_name',
                     'categories_tags',
                     'countries_tags',
                     'pnns_groups_2',
                     'energy-kcal_100g',
                     'fat_100g',
                     'saturated-fat_100g',
                     'nutrition-score-fr_100g',
                     'nutriscore_grade',
                     'ecoscore_score_fr',
                     'ecoscore_grade_fr',
                     'fiber_100g',
                     'proteins_100g',
                     'carbohydrates_100g']
    df1 = df1[selected_cols]
    
    # Restriction aux produits français
    df_app1 = df1.loc[df1.countries_tags=="en:france"]
    
    # Restriction aux "One dish meal", "Fruits" et "Yoghourts"
    cat = ['One-dish meals','Fruits', 'Milk and yogurt', 'fruits']
    df_app1 = df_app1.loc[df_app1.pnns_groups_2.isin(cat)]
    
    # Sélection des produits "français" - en réalité, on filtre pour ne garder que des product_name français.
    df_app1.drop(columns="countries_tags", inplace=True)
    
    ########## Conversion des valeurs (g) en valeurs (kcal) pour les macronutriments, et calcul de l'énergie totale apportée
    df_app1 = weight_to_energy(df)
    
    ########## Sélection des valeurs cohérentes (énergie calculée ~ énergie tabulée), uniquement #################
    ########## sur les lignes pour lesquelles les données des macronutriments sont totalement renseignées ########
    df_app1_no = clean(df_app1)
    
    ###### fusion des pnns_groups "Fruits" et "fruits"
    df_app1_no.loc[:, "pnns_groups_2"] = df_app1_no["pnns_groups_2"].apply(lambda x: "Fruits" if x=='fruits' else x)

    return df_app1_no