from fastapi import FastAPI
import uvicorn
from Model import HomeCreditDefaultModel, HomeCreditDefaultClient, Response
import numpy as np

# 2. Create app and model objects
app = FastAPI()
model = HomeCreditDefaultModel()


@app.post('/predict', response_model=Response)
def predict_default(client: HomeCreditDefaultClient):
    client_data = client.dict()
    prediction, probability = model.predict_default(list(client_data.values()))
    result = {'prediction': prediction, 'probability': probability}
    return result


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)