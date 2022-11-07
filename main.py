import uvicorn
from fastapi import FastAPI
from BankNotes import BankNote
import numpy as np
import pandas as pd
import pickle
import time
# create an app
app = FastAPI()
pickle_in = open("model.pkl","rb")
model = pickle.load(pickle_in)

@app.get('/')
def index():
    return {'message':'Hello, World'}

# Route with a single parameter, returns the parameter within a message
@app.get('/{name}')
def get_name(name:str):
    return {'Welcome: ',f'{name}'}

#3. Expose the prediction functionality, make a prediction
@app.post('/predict')
async def predict_banknote(data:BankNote):
    data = data.dict()
    print(data)
    variance = data['variance']
    skewness = data['skewness']
    curtosis = data['curtosis']
    entropy = data['entropy']
    # print(model.predict[[variance, skewness, curtosis, entropy]])
    prediction = model.predict([[variance, skewness, curtosis, entropy]])
    if(prediction[0]>0.5):
        prediction="Fake Note"
    else:
        prediction = "It is a bank note"

    return {
        'prediction':prediction
    }
    
if __name__ == "__main__":
    uvicorn.run(app, host='localhost',port=8000)