from contextlib import asynccontextmanager

import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI

MODEL_PATH = "ml/saved_model/model.joblib"


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading ML model...")

    app.state.model = joblib.load(MODEL_PATH)

    print("ML model loaded successfully.")

    yield

    print("Shutting down API...")


app = FastAPI(
    title="Iris ML Prediction API",
    lifespan=lifespan
)


@app.get("/")
def home():
    return {"message": "Iris ML Prediction API is running"}


@app.post("/predict")
def predict():
    features = pd.DataFrame([{
        "sepal length (cm)": 5.1,
        "sepal width (cm)": 3.5,
        "petal length (cm)": 1.4,
        "petal width (cm)": 0.2
    }])

    model = app.state.model
    prediction = model.predict(features)

    return {
        "prediction": str(prediction[0])
    }