from contextlib import asynccontextmanager

from pathlib import Path
import uuid

import joblib
import pandas as pd
from fastapi import FastAPI

from app.models.schemas import PredictionInput

MODEL_PATH = Path(__file__).resolve().parent.parent / "ml" / "saved_model" / "model.joblib"

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

@app.get("/health")
def health():
    model_loaded = getattr(app.state, "model", None) is not None

    return {
        "status": "ok",
        "model_loaded": model_loaded
    }

@app.post("/predict")
def predict(data: PredictionInput):
    features = pd.DataFrame([{
        "sepal length (cm)": data.sepal_length,
        "sepal width (cm)": data.sepal_width,
        "petal length (cm)": data.petal_length,
        "petal width (cm)": data.petal_width
    }])

    model = app.state.model
    prediction = model.predict(features)[0]

    confidence = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)
        confidence = float(probabilities.max())

    request_id = str(uuid.uuid4())

    return {
        "prediction": str(prediction),
        "confidence": confidence,
        "request_id": request_id
    }