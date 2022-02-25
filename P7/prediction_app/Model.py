from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')
import json
import pandas as pd
from pydantic import BaseModel, create_model
import joblib

with open('config.json', 'r') as f:
    config = json.load(f)
    
PATH = config["PATH"]

fname = PATH+'df_test.csv'
df = pd.read_csv(fname, nrows=1)
df_fname = fname
print('Dataset {} loaded'.format(df_fname))

features = df.drop(['SK_ID_CURR', 'TARGET'], axis=1, errors='ignore').columns
f = {k:(float, 0) for k in features}
HomeCreditDefaultClient = create_model('HCDCModel', **f)

class PredictResponse(BaseModel):
    prediction: int
    probability: float

class HomeCreditDefaultModel:
    def __init__(self):
        self.model_fname_ = PATH+'fitted_lgbm.pkl'
        try:
            self.model = joblib.load(self.model_fname_)
            print('Model loaded:', self.model)
        except Exception as _:
            print('You need to train the model first!')
            
    #   Make a prediction based on the user-entered data
    def predict(self, client_features):
        data_in = [client_features]
        prediction = self.model.predict(data_in)
        probability = self.model.predict_proba(data_in)[0][1]
        return prediction[0], probability