from distutils.command.clean import clean
import pandas as pd
import numpy as np
from contextlib import contextmanager
import time
import pickle
import json

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import statsmodels.api as sm
import statsmodels.formula.api as sm_api
from scipy.stats import chi2_contingency

from lightgbm import LGBMClassifier
import lightgbm_with_simple_features as lgbmsf

import shap

with open('config.json', 'r') as f:
    config = json.load(f)

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))


def anova(df, quant_var, cat_var, subcat=None):
    """Analysis of Variance"""

    model = '{} ~ {}'.format(quant_var, cat_var)
    try:
        mod = sm_api.ols(model, data=df[df[cat_var].isin(subcat)].dropna(subset=[quant_var, cat_var], how="any")).fit()           
    except:
        mod = sm_api.ols(model, data=df.dropna(subset=[quant_var, cat_var], how="any")).fit()           
    aov_table = sm.stats.anova_lm(mod, typ=2)
    eta_sq = aov_table['sum_sq'][0]/(aov_table['sum_sq'][0]+aov_table['sum_sq'][1])
    p_value = aov_table.iloc[0]["PR(>F)"]
    F_score = aov_table.iloc[0]["F"]

    #print("Résultats de l'ANOVA pour les variables {} et {}:".format(quant_var, cat_var))
    #print("F: {:.2f}, p: {:.3f}".format(aov_table.iloc[0]["F"], aov_table.iloc[0]["PR(>F)"]))
    #print("Eta-squared: {:.4f}".format(esq_sm))
    
    return F_score, p_value, eta_sq


def chi2_test(Var1, Var2, df, verbose=0):
    """Calcule les statistiques du chi2 entre les variables Var1 et Var2."""
    
    X=Var1
    Y=Var2
    cont = df[[X,Y]].pivot_table(index=X,columns=Y,aggfunc=len,margins=True,margins_name="Total")
    c = cont.fillna(0) # On remplace les valeurs nulles par 0
    chi2, pval, dof, chi2_table = chi2_contingency(c.iloc[:-1, :-1])
    if verbose > 0:
        print("coefficient du chi2: {:.0f}".format(chi2))
        print("p-valeur: {}".format(pval))
    return chi2, pval


def contingence(Var1, Var2, df):
    """Calcule et affiche la matrice de contingence entre les variables Var1 et Var2
    sous forme de heatmap"""
    
    X=Var1
    Y=Var2
    cont = df[[X,Y]].pivot_table(index=X,columns=Y,aggfunc=len,margins=True,margins_name="Total")

    tx = cont.loc[:,["Total"]]
    ty = cont.loc[["Total"],:]
    n = len(df)
    indep = tx.dot(ty) / n

    c = cont.fillna(0) # On remplace les valeurs nulles par 0
    measure = (c-indep)**2/indep
    xi_n = measure.sum().sum()
    table = measure/xi_n
    
    plt.figure(figsize=(18, 6))
    sns.heatmap(table.iloc[:-1,:-1],annot=c.iloc[:-1,:-1], fmt=".0f")


def dropna_cols(df, perc_filled):
    """Retire les colonnes remplies à moins de perc_filled. perc_filled est compris entre 0 et 1."""
    nna = df.notna().sum()/df.shape[0]
    nfilled_cols = nna[nna<perc_filled].index
    cleaned_df = df.drop(columns=nfilled_cols)
    
    return cleaned_df


def impute(df, iterative=False, **kwargs):
    """Réalise l'imputation des valeurs manquantes. Deux méthodes disponibles au choix via le booléen iterative:
    - SimpleImputer
    - IterativeImputer
    Dans les deux cas, la méthode d'imputation ou l'estimateur choisi peuvent être passés via les kwargs. Voir les
    documentations correspondantes pour la syntaxe est les choix possibles.
    """
    #num_cols = [col for col in df.columns if df[col].dtype!='object']
    iterative = kwargs.pop('iterative', False)
    if not iterative:
        strategy = kwargs.pop('strategy', None)
        imp = SimpleImputer(missing_values=np.nan, strategy=strategy)
        imp.fit(df)
        filled_df = imp.transform(df)
            
    else:
        estimator = kwargs.pop('estimator', None)
        imp = IterativeImputer(missing_values=np.nan, estimator=estimator)
        imp.fit(df)
        filled_df = imp.fit_transform(df)
            
    return pd.DataFrame(data=filled_df, columns=df.columns)
    
    
def cleanup(df, perc_filled, imput=False, **kwargs):
    """Réalise le nettoyage des colonnes vides et l'imputation en cascade"""
    
    cleaned_df = dropna_cols(df, perc_filled)
    if imput:
        filled_df = impute(cleaned_df, **kwargs)
    else:
        filled_df = cleaned_df.dropna(how='any')
    return filled_df


def create_datasets(num_rows=None, save=True, path_to_save='./sample_data/', path_to_read='./files/data/'):
    """Réalise l'importation des tables depuis path_to_read,le prétraitement (cleanup) et 
    enregistre les jeux de données créés dans le dossier désigné par path_to_save"""
    
    print('prepare data...')
    # création d'un echantillon pour le dashboard
    df = lgbmsf.join_df(num_rows=num_rows)

    # On retire les colonnes ayant moins de 0% de leurs valeurs renseignées
    cleaned_df = dropna_cols(df, 0.7)
    cleaned_df.replace([np.inf, -np.inf], np.nan, inplace=True)

    one_hot_cols = [col for col in cleaned_df.columns if cleaned_df[col].nunique()<=2]
    cont_cols = [col for col in cleaned_df if col not in one_hot_cols]

    filled_data_cont = impute(cleaned_df[cont_cols], iterative=False, strategy='mean')
    filled_data_ohc = impute(cleaned_df[one_hot_cols], iterative=False, strategy='median')

    filled_data = pd.concat([filled_data_cont, filled_data_ohc], axis=1)
    #filled_data.index = pd.Index(range(filled_data.shape[0]))

    # Récupération des clients présents dans le jeu de données traité
    skidcurr = filled_data.SK_ID_CURR.unique()
    
    # Création des tables à partir des clients sélectionnés
    print('Processing sample files...')
    for i in [
    'POS_CASH_balance.csv',
    'credit_card_balance.csv',
    'installments_payments.csv',
    'application_train.csv',
    'bureau.csv',
    'previous_application.csv',
    ]:
        with timer("Process "+i):
            df = pd.read_csv(path_to_read+i)
            df = df[df.SK_ID_CURR.isin(skidcurr)]
            print(i.split('.')[0]+" df shape:", df.shape)
            #df.index = pd.Index(range(df.shape[0]))
            df.to_csv(path_to_save+i, index_label=False)

    skidbureau = pd.read_csv(path_to_save+'bureau.csv').SK_ID_BUREAU
    bb = pd.read_csv(path_to_read+'bureau_balance.csv')
    bb = bb[bb.SK_ID_BUREAU.isin(skidbureau)]
    #bb.index = pd.Index(range(bb.shape[0]))
    print("bureau_balance df shape:", bb.shape)
    file_name = 'bureau_balance.csv'
    bb.to_csv(path_to_save+file_name, index_label=False)
    print("Done")

    if save:
        # enregistre filled_data 
        with timer('Saving cleaned dataset to {}...'.format(path_to_save+'filled_data.csv')):
            filled_data.to_csv(path_to_save+'filled_data.csv', index_label=False)
    
    return filled_data

def prepare_data(df=None, path=None):
    """Prépare les jeux d'entraînement et de test à partir de l'échantillon en vue de 
    l'entraînement du modèle"""

    if path:
        # chargement du dataset
        with timer('Loading data from {}'.format(path+'filled_data.csv')):
            df = pd.read_csv(path+'filled_data.csv')
    else:
        try:
            # train-test
            X = df.drop(columns=['TARGET'])
            y = df.TARGET
        except:
            print("You must provide a path or a dataset")

    # Séparation en jeux d'entraînement/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

    # Sauvegarde du jeu de test
    X_test.index = pd.Index(range(X_test.shape[0]))
    with timer('Saving test data at {}'.format(path+'df_test.csv')):
        X_test.to_csv(path+'df_test.csv', index_label=False)
    
    return X_train, X_test, y_train, y_test


def run_model(df_train, df_test, y_train, y_test, path='./sample_data/'):
    """Réalise l'entraînement du modèle (LightGBM)."""
    
    best_params = {'colsample_bytree': 0.5, 'max_depth': -1, 'n_estimators': 100, 'num_leaves': 10}
    bestmodel = LGBMClassifier(early_stopping_round=200,
                                objective='binary',
                                metric='AUC',
                                is_unbalance=True,
                                silent=False,
                                verbosity=-1,
                            **best_params)

    X_train = df_train.drop('SK_ID_CURR', axis=1).to_numpy()
    X_test = df_test.drop('SK_ID_CURR', axis=1).to_numpy()
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    bestmodel.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=-1)

    # On sauvegarde le meilleur modèle
    with open(path+'fitted_lgbm.pkl', 'wb') as f:
        pickle.dump(bestmodel, f)

    # Performance sur le jeu d'entraînement
    #print("\nMeilleure performance sur le jeu d'entraînement/validation' : {:.3f}".format(regr.best_score_))
    print("Performance sur le jeu d'entraînement' : {:.3f}".format(bestmodel.score(X_train, y_train)))

    # Performance en généralisation du meilleur modèle sur le jeu de test
    #y_pred = regr.predict(X_test)
    y_pred = bestmodel.predict(X_test)
    print("Performance en généralisation sur le jeu de test : {:.3f}".format(roc_auc_score(y_test, y_pred)))
    print("Esperance de la variable cible: {}".format(bestmodel.predict_proba(X_test).mean(0).round(3)))
    return bestmodel


def create_shap_data(model, path='./sample_data/'):
    X = pd.read_csv(path+'filled_data.csv')
    bg_data = X.drop(['SK_ID_CURR', 'TARGET'], axis=1)
    tree_explainer = shap.TreeExplainer(model,
                                     data=bg_data,
                                     feature_perturbation='interventional',
                                     model_output='probability'
                                    )
    X_test = pd.read_csv(path+'df_test.csv')
    shap_values = tree_explainer(X_test.drop(['SK_ID_CURR'], axis=1))
    base_value = tree_explainer.expected_value
    shap_values.values = shap_values.values.astype('float32')
    print(base_value)
    with open(path+'shap_values.pkl', 'wb') as f:
        pickle.dump(shap_values.values, f)
    
    with open(path+'base_value.pkl', 'wb') as f:
        pickle.dump(base_value, f)

def main(num_rows=None):
    print('Creating datasets')
    filled_data = create_datasets(num_rows=num_rows, save=True, path_to_save=config['SAVE_TO'], path_to_read=config['READ_FROM'])
    print('Preparing data')
    X_train, X_test, y_train, y_test = prepare_data(filled_data)
    print('Training model...')
    model = run_model(X_train, X_test, y_train, y_test)
    print('Calculating shap values')
    create_shap_data(model)

if __name__ == '__main__':
    main(num_rows=10000)