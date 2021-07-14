import pandas as pd
import numpy as np

def load_select(dataset, selected_cols, restr="en:france"):
    '''Fonction qui prend en entrée un jeu de données dataset au format csv, et renvoie un dataframe ne contenant
    que les variables sélectionnées

    dataset: chemin vers le dataset
    selected_cols: liste des colonnes à conserver pour analyse
    restr: valeur de countries_tags dans le cas d'un filtrage sur cette variable. "en:france" par défaut.'''
    
    # 1. Chargement du jeu de données
    df = pd.read_csv(dataset, sep='\t', low_memory=False)
    
    # 2. Suppression des lignes pour lesquelles le product name n'est pas renseigné
    df1 = df.copy()
    df1.dropna(subset=["product_name"], inplace=True)

    
    # 3. sélection des variables
    df1 = df1[selected_cols]
    
    # 4. Restriction aux produits français    
    try:
        df_app1 = df1.loc[df1.countries_tags==restr]
    except:
        df_app1 = df1

    # 5. # On enlève la colonne "countries_tags" du dataframe, elle ne nous est plus utile.
    df_app1.drop(columns="countries_tags", inplace=True)
    
    return df_app1


def weight_to_energy(df, start_col, new_col, coef):
    '''Prend en entrée un dataframe df, convertit les valeurs (g) en valeurs (kcal) pour les macronutriments, et 
    calcule l'énergie totale apportée par ces macronutriments. L'énergie calculée pour chaque produit est contenue dans une
    nouvelle colonne "total_energy_from_nutriments'''

    # Création de la colonne new_col à partir de la colonne start_colet du coefficient coef
    df.loc[:, new_col] = df[start_col].apply(lambda x: x*coef)
    
    # df.loc[:, "carbohydrates_kcal"] = df["carbohydrates_100g"].apply(lambda x: x*4)
    # df.loc[:, "proteins_kcal"] = df["proteins_100g"].apply(lambda x: x*4)
    # df.loc[:, "fiber_kcal"] = df["fiber_100g"].apply(lambda x: x*1.9)

 
    # # Calcul de la somme dans une nouvelle colonne "total_energy_from_nutriments"
    # df.loc[:, "total_energy_from_nutriments"] = df[["fat_kcal","carbohydrates_kcal","proteins_kcal", "fiber_kcal"]].apply(lambda x: np.sum(x), axis=1)
    
    # ########## Correction des valeurs données en kJ mais indiquées comme kcal. #########################
    # # 1. Sélection des valeurs concernées
    # df.loc[:, "w_kcal"] = df[["total_energy_from_nutriments", "energy-kcal_100g"]].apply(lambda x: True if (x[0]<0.28*x[1] 
    #                                                                      and x[0]>0.19*x[1]) else False, axis=1)
    # # 2. Correction
    # df.loc[:, "energy-kcal_100g"] = df[["energy-kcal_100g", "w_kcal"]].apply(lambda x: x[0]*0.239 if x[1] else x[0], axis=1)
    # df.drop(columns="w_kcal", inplace=True)

    return df


def joule_to_kcal(df, var_to_convert, reference_var, cut_off_high, cut_off_low):
    '''Multiplie les valeurs associées à la colonne var_to_convert du dataframe df par 0.239 (1 kcal = 0.239 kJ) à condition que la valeur se trouve dans un intervalle
    centré sur la valeur de reference_var, et de limites hautes et basses cut_off_high, cut_off_low'''
    
    # Ajout d'une colonne convert de type booléen (True/False). True signifie que le point est éligible à la conversion.
    # False signifie que le point est aberrant et n'est donc pas inclus dans le traitement.
    df.loc[:, convert] = df[[reference_var, var_to_convert]].apply(lambda x: True if (x[0]<cut_off_high*x[1] 
                                                                          and x[0]>cut_off_low*x[1]) else False, axis=1)
    
    # Application de la conversion aux points éligibles.
    df.loc[:, var_to_convert] = df[[var_to_convert, convert]].apply(lambda x: x[0]*0.239 if x[1] else x[0], axis=1)
    
    # Nettoyage du dataframe
    df.drop(columns="ratio", inplace=True)
    
    return df


def make_mask(df, target_cols, method, loc_mask=None):
    '''Crée un masque sur le dataframe. Pour créer le masque, deux méthodes possibles: une vérification 
    que toutes les colonnes target_cols sont renseignées (isna().sum()==0), ou une vérification que 
    target_cols[0]>target_cols[1].
    df: pandas.DataFrame
    target_cols: liste des colonnes sur les quelles la méthode method est appliquée
    method:"greaterthan", "isna"
    loc_mask: pd.DataFrame. Masque pour sélectionner une portion du df sur lequel appliquer method''' 
    
    df1 = df.copy()
    
    if loc_mask is None:
        loc_mask = df1
    
    if method=="greaterthan":
        new_mask = df.loc[loc_mask, target_cols].apply(lambda x: True if x[0]>x[1] else False, axis=1)
    elif method=="isna":
        new_mask = df.loc[loc_mask, target_cols].apply(lambda x: True if x.isna().sum()==0 else False, axis=1)
    
    # Mise à jour du dataframe
    df1.loc[:, "complete_vars"] = e_nn
    
    return df1


def replace_with_nan(df, target_col):
    '''Remplace les valeurs nulles de la colonne target_col du dataframe df par des NaN'''
    df.loc[:, [target_col]] = df[target_col].apply(lambda x: np.nan if x==0 else x)
    return df


def create_distrib(df, col1="energy-kcal_100g", col2="total_energy_from_nutriments", reverse_cols=False, method="diff", mask=None):
    '''Crée une distribution à partir des colonnes "total_energy_from_nutriments" et "energy-kcal_100g".
    Selon la méthode retenue, "diff" ou "ratio", la fonction renvoie la différence ou le quotient associé à ces deux variables
    sous forme d'une Series portant le nom de la méthode retenue.
    df: pd.DataFrame
    reverse_cols: bool, par défaut:False. Si True, intervertit les deux colonnes utilisées pour contruire la distribution
    mask: "None", "notna" ou "e_nn". Valeur par défaut: "None"'''

   
    if reverse_cols:
        col2 = "energy-kcal_100g"
        col1 = "total_energy_from_nutriments"

    if mask==None:
        loc_ind1 = df
        loc_ind2 = df
    elif mask=="notna":
        loc_ind1 = df[col1].notna()
        loc_ind2 =  df[col2].notna()
    elif mask=="e_nn":
        loc_ind1 = df["e_nn"]==True
        loc_ind2 = df["e_nn"]==True

    s1 = df.loc[loc_ind1, col1]
    s2 = df.loc[loc_ind2, col2]
    
    if method=="diff":        
        distrib = pd.Series(s1-s2, name="diff")

    elif method=="ratio":
        distrib = pd.Series(s1/s2, name="ratio")
    
    return distrib

def assign_grade(x):
    if x<-1 and x>=-15:
        grade = "a"
    elif x>=-1 and x<4:
        grade = "b"
    elif x>=4 and x<12:
        grade = "c"
    elif x>=12 and x<17:
        grade = "d"
    elif x>=17 and x<40:
        grade = "e"
    else:
        grade = np.nan
    return grade

