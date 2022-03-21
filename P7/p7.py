from distutils.command.clean import clean
import pandas as pd
import numpy as np
from contextlib import contextmanager
import time
import pickle
import json
import gc

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report

import statsmodels.api as sm
import statsmodels.formula.api as sm_api
from scipy.stats import chi2_contingency

from lightgbm import LGBMClassifier
import lightgbm_with_simple_features as lgbmsf

import shap

with open('config.json', 'r') as f:
    config = json.load(f)

with open('model_params.json', 'r') as f:
    model_params = json.load(f)
    
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

def drop_cols(df, threshold=.8, sample_frac=.1):
    """Retire une colonne parmi chaque paire de colonnes (col1, col2) présentant une corrélation supérieure
    à perc_filled."""

    sample_df = df.sample(frac=sample_frac)
    
    with timer("Computing features correlation matrix"):
        corr = sample_df.drop('SK_ID_CURR', axis=1).corr().abs()
        
    # triangle supérieur des corrélations
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype('bool'))

    # Sélection des colonnes au-dessus du seuil de corrélation
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    cleaned_df = df.drop(columns=to_drop)

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


def create_datasets(load_from_existing=True, num_rows=None, sample_fraction=.25, save=True, path_to_read=config['READ_FROM']):
    """Réalise l'importation des tables depuis path_to_read,le prétraitement (cleanup) et 
    enregistre les jeux de données créés dans le dossier désigné par path_to_save"""
    
    print('prepare data...')
    if load_from_existing:
        try:
            cleaned_df = pd.read_csv(path_to_read+'data.csv')
        except:
            print("No dataset found! Try setting load_from_existing to False, or check the filepath is correct.")

    else:
        # création du dataset complet
        df = lgbmsf.join_df(num_rows=num_rows)

        # On retire une feature pour chaque paire de features corrélées à plus de 80%
        cleaned_df = drop_cols(df, sample_frac=sample_fraction)
        cleaned_df.replace([np.inf, -np.inf], np.nan, inplace=True)

        if save:
            # enregistre sample_data pour utilisation dans le dashboard 
            with timer('Saving cleaned dataset to {}...'.format(path_to_read+'data.csv')):
                cleaned_df.to_csv(path_to_read+'data.csv', index_label=False)
    
    return cleaned_df

def prepare_data(df=None, test_size=.05, path_to_save=config['SAVE_TO'], path_to_read=config['READ_FROM']):
    """Prépare les jeux d'entraînement et de test à partir de l'échantillon en vue de 
    l'entraînement du modèle"""

    features_to_keep = model_params['most_important_features']

    if df is None:
        try:
            with timer('Loading data from {}'.format(path_to_read+'data.csv')):
                    df = pd.read_csv(path_to_read+'data.csv')
                
        except:
            print("You must provide a path or a dataset")
    
    X = df[features_to_keep + ['SK_ID_CURR']]
    y = df.TARGET

    # Séparation en jeux d'entraînement/test. On ne garde que 10% du jeu total pour le jeu de test
    # en vue du déploiement via heroku qui impose des limites sur la taille des données
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y)
    print("Train data shape: {}".format(X_train.shape))
    print("Test data shape: {}".format(X_test.shape))

    # Récupération des clients présents dans le jeu de test
    skidcurr = X_test.SK_ID_CURR.unique()

    # Création des tables à partir des clients sélectionnés pour affichage dans le dashboard
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
            df.to_csv(path_to_save+i, index_label=False)

    skidbureau = pd.read_csv(path_to_save+'bureau.csv').SK_ID_BUREAU
    bb = pd.read_csv(path_to_read+'bureau_balance.csv')
    bb = bb[bb.SK_ID_BUREAU.isin(skidbureau)]
    print("bureau_balance df shape:", bb.shape)
    file_name = 'bureau_balance.csv'
    bb.to_csv(path_to_save+file_name, index_label=False)
    print("Done")

    # Sauvegarde du jeu de test
    X_test.index = pd.Index(range(X_test.shape[0]))
    with timer('Saving test data at {}'.format(path_to_save+'features_test.csv')):
        X_test.to_csv(path_to_save+'features_test.csv', index_label=False)
    
    return X_train, X_test, y_train, y_test


def run_model(features_train, features_test, y_train, y_test, path='./sample_data/'):
    """Réalise l'entraînement du modèle (LightGBM)."""
    
    best_params = model_params['best_params']
    features_to_keep = model_params['most_important_features']
    
    #X_train = features_train.drop('SK_ID_CURR', axis=1).to_numpy()
    X_train = features_train.drop('SK_ID_CURR', axis=1).to_numpy()
    #X_test = features_test.drop('SK_ID_CURR', axis=1).to_numpy()
    X_test = features_test.drop('SK_ID_CURR', axis=1).to_numpy()
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    feats = features_to_keep
    #feats = [f for f in features_train.drop('SK_ID_CURR', axis=1).columns]

    model = LGBMClassifier(early_stopping_round=50,
                                objective='binary',
                                metric='AUC',
                                #is_unbalance=True,
                                silent=False,
                                verbosity=-1,
                            **best_params)

    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], 
            eval_metric= 'auc', verbose= 200, early_stopping_rounds= 50)
    print()
    print("Entraînement avec les paramètres suivants: ")
    for k, v in best_params.items():
        print("{} = {}".format(k, v))
    print()
    print("Résultats de l'entraînement")
    print("---------------------------")
    # Performance sur le jeu d'entraînement
    print("Performance sur le jeu d'entraînement : {:.3f}".format(model.score(X_train, y_train)))
    # Performance en généralisation du meilleur modèle sur le jeu de test
    #y_pred = regr.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    print("Performance en généralisation sur le jeu de test : {:.3f}".format(roc_auc_score(y_test, y_pred_proba)))
    print("Espérance de la variable y (jeu de test): {}".format(y_pred.mean()))
    print("Espérance de la probabilité de faire défaut (jeu de test): {}".format(y_pred_proba.mean(0).round(3)[1]))
    print()
    print("Classification Report")
    print(classification_report(y_test, model.predict(X_test)))

    ############## Feature importances ################

    feature_importance_df = pd.DataFrame()
    feature_importance_df["feature"] = feats
    feature_importance_df["importance"] = model.feature_importances_
    
    # Threshold for cumulative importance
    threshold = 0.99
    #norm_feature_importances = plot_feature_importances(feature_importance_df, threshold=threshold)
    plot_feature_importances(feature_importance_df, threshold=threshold)
    # Extract the features to keep
    #features_to_keep = list(norm_feature_importances[norm_feature_importances['cumulative_importance'] < threshold]['feature'])

    # Sauvegarde du modèle
    with open(path+'lgb.pkl', 'wb') as f:
        pickle.dump(model, f)

    # Sauvegarde des features
    #model_params['most_important_features'] = features_to_keep
    #with open("model_params.json", "w") as f:
    #    json.dump(model_params, f)

    return model

# Display/plot feature importance
def plot_feature_importances(df, threshold = 0.9):
    """
    Plots 15 most important features and the cumulative importance of features.
    Prints the number of features needed to reach threshold cumulative importance.
    
    Parameters
    --------
    df : dataframe
        Dataframe of feature importances. Columns must be feature and importance
    
    threshold : float, default = 0.9
        Threshold for prining information about cumulative importances
        
    Return
    --------
    df : dataframe
        Dataframe ordered by feature importances with a normalized column (sums to 1)
        and a cumulative importance column
    
    """
    
    plt.rcParams['font.size'] = 18
    
    # Sort features according to importance
    df = df.sort_values('importance', ascending = False).reset_index()
    
    # Normalize the feature importances to add up to one
    df['importance_normalized'] = df['importance'] / df['importance'].sum()
    df['cumulative_importance'] = np.cumsum(df['importance_normalized'])

    # Make a horizontal bar chart of feature importances
    fig, ax = plt.subplots(figsize = (12, 6))
    fig.subplots_adjust(left=.4, right=.99)
    # Need to reverse the index to plot most important on top
    ax.barh(list(reversed(list(df.index[:15]))), 
            df['importance_normalized'].head(15), 
            align = 'center', edgecolor = 'k')
    
    # Set the yticks and labels
    ax.set_yticks(list(reversed(list(df.index[:15]))))
    ax.set_yticklabels(df['feature'].head(15))
    
    # Plot labeling
    plt.xlabel('Normalized Importance'); plt.title('Feature Importances')
    plt.draw()
    
    # Cumulative importance plot
    #plt.figure(figsize = (8, 6))
    #plt.plot(list(range(len(df))), df['cumulative_importance'], 'r-')
    #plt.xlabel('Number of Features')
    #plt.ylabel('Cumulative Importance')
    #plt.title('Cumulative Feature Importance')
    #plt.draw()
    
    #importance_index = np.min(np.where(df['cumulative_importance'] > threshold))
    #print('>>> %d features required for %0.2f of cumulative importance <<<' % (importance_index + 1, threshold))
    #print()
    #return df

def create_shap_data(model, path_to_save=config['SAVE_TO'], path_to_read=config['READ_FROM']):
    """Calculate SHAP values for the model"""

    X = pd.read_csv(path_to_read+'data.csv')
    #bg_data = X.drop(['SK_ID_CURR', 'TARGET'], axis=1).sample(10000)
    features_to_keep = model_params['most_important_features']
    bg_data = X[features_to_keep].sample(10000)
    tree_explainer = shap.TreeExplainer(model,
                                     data=bg_data,
                                     feature_perturbation='interventional',
                                     model_output='probability'
                                    )
    X_test = pd.read_csv(path_to_save+'features_test.csv')
    shap_values = tree_explainer(X_test.drop('SK_ID_CURR', axis=1))
    base_value = tree_explainer.expected_value
    shap_values.values = shap_values.values.astype('float32')
    print("SHAP base value (espérance de la probabilité de faire défaut): {:.3f}".format(base_value))
    with open(path_to_save+'shap_values.pkl', 'wb') as f:
        pickle.dump(shap_values.values, f)
    
    with open(path_to_save+'base_value.pkl', 'wb') as f:
        pickle.dump(base_value, f)

def main(num_rows=10000, sample_fraction=.1, test_size=.05):
    print('Creating datasets')
    data = create_datasets(num_rows=num_rows, sample_fraction=sample_fraction)
    print('Preparing data')
    X_train, X_test, y_train, y_test = prepare_data(data, test_size=test_size)
    print('Training model...')
    model = run_model(X_train, X_test, y_train, y_test)
    print('Calculating shap values')
    create_shap_data(model)
    plt.show()

if __name__ == '__main__':
    main(num_rows=None, sample_fraction=.25, test_size=.05)