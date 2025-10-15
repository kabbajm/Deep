from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel

app = FastAPI(title="Iris Classification API")
model = joblib.load("artifacts/model.joblib")

class IrisData(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/")
def root():
    return {"message": "API prête à prédire"}

@app.post("/predict")
def predict(data: IrisData):
    X = np.array([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])
    pred = model.predict(X)[0]
    prob = model.predict_proba(X).max()
    return {"prediction": int(pred), "probability": float(prob)}