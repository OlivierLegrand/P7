from fastapi import FastAPI, Response
from typing import List
import uvicorn
from Model import HomeCreditDefaultModel, HomeCreditDefaultClient, PredictResponse
import numpy as np
import pandas as pd
import joblib
import json

with open('config.json', 'r') as f:
    config = json.load(f)
    
PATH = config["PATH"]

# Create app and model objects
app = FastAPI()
model = HomeCreditDefaultModel()

# load shap values
print("Loading shap values from directory {}".format(PATH))
shap_values = joblib.load(open(PATH+'shap_values.pkl', 'rb'))
base_value = joblib.load(open(PATH+'base_value.pkl', 'rb'))

# load column definitions
hc_columns_definitions = pd.read_csv('../HomeCredit_columns_description.csv')[['Row', 'Description']]

@app.post('/predict/', response_model=PredictResponse)
def predict_default(client: HomeCreditDefaultClient):
    print(type(client))
    client_data = client.dict()
    prediction, probability = model.predict(list(client_data.values()))
    result = {'prediction': prediction, 'probability': probability}
    return result

@app.post('/shap_values/')
def return_shap_values(idx:List=[]):
    shap_vals = shap_values[idx]
    return shap_vals.tolist()

@app.get('/base_value/')
def return_base_value():
    return base_value

@app.post('/definition/')
def return_column_definition(column:str):
    col_analyzer = column.split('_')
    prefix = col_analyzer[0]
    suffix = col_analyzer[-1]
    if prefix in ['BURO', 'PREV', 'ACTIVE', 'CLOSED', 'APPROVED', 'REFUSED', 'POS', 'INSTAL', 'CC']:
        if suffix in ['MEAN', 'MAX', 'MIN', 'VAR', 'SUM', 'SIZE']:
            col_name = '_'.join(col_analyzer[1:-1])
        else:
            col_name = '_'.join(col_analyzer[1:])
    else:
        col_name = '_'.join(col_analyzer)
    return hc_columns_definitions.loc[hc_columns_definitions['Row']==col_name, 'Description'].to_numpy()[0]

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)