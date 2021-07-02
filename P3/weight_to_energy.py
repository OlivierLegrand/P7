import pandas as pd
import numpy as np

def weight_to_energy(df):
'''Prend en entrée un dataframe df, convertit les valeurs (g) en valeurs (kcal) pour les macronutriments, et 
calcule l'énergie totale apportée par ces macronutriments. L'énergie calculée pour chaque produit est contenue dans une
nouvelle colonne "total_energy_from_nutriments'''

    # Création des colonnes fat_kcal, carbohydrates_kcal, proteins_kcal, fiber_kcal et total_energy_from_nutriments
    df[:, "fat_kcal"] = df["fat_100g"].apply(lambda x: x*9)
    df.loc[:, "carbohydrates_kcal"] = df["carbohydrates_100g"].apply(lambda x: x*4)
    df.loc[:, "proteins_kcal"] = df["proteins_100g"].apply(lambda x: x*4)
    df.loc[:, "fiber_kcal"] = df["fiber_100g"].apply(lambda x: x*1.9)
    
    # Calcul de la somme dans une nouvelle colonne "total_energy_from_nutriments"
    df.loc[:, "total_energy_from_nutriments"] = df[["fat_kcal","carbohydrates_kcal","proteins_kcal", "fiber_kcal"]].apply(lambda x: np.sum(x), axis=1)
    
    ########## Correction des valeurs données en kJ mais indiquées comme kcal. #########################
    # 1. Sélection des valeurs concernées
    df.loc[:, "w_kcal"] = df[["total_energy_from_nutriments", "energy-kcal_100g"]].apply(lambda x: True if (x[0]<0.28*x[1] 
                                                                         and x[0]>0.19*x[1]) else False, axis=1)
    # 2. Correction
    df.loc[:, "energy-kcal_100g"] = df[["energy-kcal_100g", "w_kcal"]].apply(lambda x: x[0]*0.239 if x[1] else x[0], axis=1)
    df.drop(columns="w_kcal", inplace=True)

    return df