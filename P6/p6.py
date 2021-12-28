"""Custom functions for P6 project"""

from ast import literal_eval
import pandas as pd
from sklearn import metrics

# visualisation des categories dans un dataframe
def category_trees(data):
    
    prod_cat_lists = []
    product_trees = (data["product_category_tree"].apply(lambda x: x.replace(' >> ', '","'))
                     .apply(literal_eval)
                    )
    
    for tree in product_trees:
        prod_cat_lists.append(tree)
    
    df = pd.DataFrame(prod_cat_lists)
    
    
    return df


# création de la variable des catégories de produits
def extract_categories_from_tree(data, level=0):
    
    df = category_trees(data)

    return df[level]

def load_data(path=None):
    """docstring"""
    
    if path is None:
        rootpath = "./data/Flipkart/"
        path = rootpath + "flipkart_com-ecommerce_sample_1050.csv"
        data = pd.read_csv(path)
    
    else:
        data = pd.read_csv(path)

    return data

def conf_mat_transform(y_true,y_pred, corresp) :
    conf_mat = metrics.confusion_matrix(y_true,y_pred)
    
    #corresp = np.argmax(conf_mat, axis=0)
    #corresp = [3, 1, 2, 0]
    print ("Correspondance des clusters : ", corresp)
    # y_pred_transform = np.apply_along_axis(correspond_fct, 1, y_pred)
    labels = pd.Series(y_true, name="y_true").to_frame()
    labels['y_pred'] = y_pred
    labels['y_pred_transform'] = labels['y_pred'].apply(lambda x : corresp[x]) 
    
    return labels['y_pred_transform']
