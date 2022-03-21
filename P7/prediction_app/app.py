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

# 2. Create app and model objects
app = FastAPI()
model = HomeCreditDefaultModel()

# load shap values
print("Loading shap values from directory {}".format(PATH))
shap_values = joblib.load(open(PATH+'shap_values.pkl', 'rb'))
base_value = joblib.load(open(PATH+'base_value.pkl', 'rb'))

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

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)