import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer

import statsmodels.api as sm
import statsmodels.formula.api as sm_api
from scipy.stats import chi2_contingency




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
    num_cols = [col for col in df.columns if df[col].dtype!='object']
    
    if not iterative:
        strategy = kwargs.pop('strategy', None)
        imp = SimpleImputer(missing_values=np.nan, strategy=strategy)
        imp.fit(df[num_cols])
        filled_df = imp.transform(df)
            
    else:
        estimator = kwargs.pop('estimator', None)
        imp = IterativeImputer(missing_values=np.nan, estimator=estimator)
        imp.fit(df[num_cols])
        filled_df = imp.fit_transform(df)
            
    return filled_df
    
    
def cleanup(df, perc_filled, iterative=False, **kwargs):
    """Réalise le nettoyage des colonnes vides et l'imputation en cascade"""
    
    cleaned_df = dropna_cols(df, perc_filled)
    filled_df = impute(cleaned_df, iterative=iterative)
    
    return filled_df
        