# 1. Library imports
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')
import json
import pandas as pd
import lightgbm_with_simple_features as lgbmsf
from lightgbm import LGBMClassifier
from pydantic import BaseModel
import joblib
#from sklearn.model_selection import train_test_split

with open('config.json', 'r') as f:
    config = json.load(f)
    
NUM_ROWS = config["NUM_ROWS"]
PATH = config["PATH"]

def clean_df(df):
    nna = df.notna().sum()/df.shape[0]
    nfilled_cols = nna[nna<0.7].index
    filled_df = df.drop(columns=nfilled_cols)
    filled_df.fillna(filled_df.mean(), inplace=True)
    return filled_df



# 2. Class which describes a single client parameters
class HomeCreditDefaultClient(BaseModel):
    client_features: List = []
    client_name: str = 'Smith'

class Response(BaseModel):
    prediction: int
    probability: float


# 3. Class for training the model and making predictions
class HomeCreditDefaultModel:
    # 6. Class constructor, loads the dataset and loads the model
    #    if exists. If not, calls the _train_model method and 
    #    saves the model
    def __init__(self):
        # Jointures
        try:
            fname = './filled_data.csv'
            self.df = pd.read_csv('./filled_data.csv')
            self.df_fname = './filled_data.csv'
            print('Dataset {} loaded'.format(self.df_fname))
        except:
            self.df = clean_df(lgbmsf.join_df(num_rows=NUM_ROWS))
    
        self.model_fname_ = 'fitted_lgbm.pickle'
        try:
            self.model = joblib.load(self.model_fname_)
            print('Model loaded:', self.model)
        except Exception as _:
            print('You need to train the model first!')
            self.model = self._train_model()
            joblib.dump(self.model, self.model_fname_)
        

    # 4. Perform model training using the LightGBM classifier
    def _train_model(self):
        X = self.df.drop(['SK_ID_CURR', 'TARGET'], axis=1)
        y = self.df['TARGET']

        bestmodel = LGBMClassifier(early_stopping_round=200,
                               objective='binary',
                               metric='AUC',
                               is_unbalance=True,
                               silent=False,
                               verbosity=-1,
                               colsample_bytree=0.5,
                               max_depth=-1, 
                               n_estimators=100,
                               num_leaves= 10)

        model = bestmodel.fit(X, y)
        return model


    # 5. Make a prediction based on the user-entered data
    #    Returns the predicted species with its respective probability
    def predict_default(self, client_features):
        # Séparation en jeux d'entraînement/test
        X = self.df.drop(['SK_ID_CURR', 'TARGET'], axis=1).to_numpy()
        data_in = [client_features]
        prediction = self.model.predict([client_features])
        probability = self.model.predict_proba([client_features]).max()
        return prediction[0], probability