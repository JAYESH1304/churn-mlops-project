from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load("models/model.pkl")


@app.get("/")
def home():
    return {"message": "Churn Prediction API"}


@app.post("/predict")
def predict(data: dict):

    df = pd.DataFrame([data])

    prediction = model.predict(df)[0]

    return {"prediction": int(prediction)}