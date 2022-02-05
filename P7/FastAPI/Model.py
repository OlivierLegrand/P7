# 1. Library imports
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import lightgbm_with_simple_features as lgbmsf
import json
import seaborn as sns
import re
import missingno as msno
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

from lightgbm import LGBMClassifier

from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV, ParameterGrid

from pydantic import BaseModel
import joblib

with open('config.json', 'r') as f:
    config = json.load(f)
    
NUM_ROWS = config["NUM_ROWS"]
PATH = config["PATH"]


# 2. Class which describes a single client parameters
class HomeCreditDefaultClient(BaseModel):
    client_parameters: list


# 3. Class for training the model and making predictions
class HomeCreditDefaultModel:
    # 6. Class constructor, loads the dataset and loads the model
    #    if exists. If not, calls the _train_model method and 
    #    saves the model
    def __init__(self):
        # Jointures
        self.df = lgbmsf.join_df(num_rows=NUM_ROWS)
        self.model_fname_ = 'fitted_lgbm.pickle'
        try:
            self.model = joblib.load(self.model_fname_)
        except Exception as _:
            print('You need to train the model first!')
            self.model = self._train_model()
            joblib.dump(self.model, self.model_fname_)
        

    # 4. Perform model training using the RandomForest classifier
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
    def predict_species(self, client_parameters):
        data_in = [client_parameters]
        prediction = self.model.predict(data_in)
        probability = self.model.predict_proba(data_in).max()
        return prediction[0], probability