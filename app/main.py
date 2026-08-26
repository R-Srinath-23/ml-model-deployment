from contextlib import asynccontextmanager

from pathlib import Path
import uuid

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from app.models.schemas import PredictionInput, PredictionOutput

MODEL_PATH = Path(__file__).resolve().parent.parent / "ml" / "saved_model" / "model.joblib"

class PredictionError(Exception):
    pass

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

@app.exception_handler(PredictionError)
async def prediction_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Prediction failed"
        }
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

@app.post("/predict", response_model=PredictionOutput)
def predict(data: PredictionInput):

    request_id = str(uuid.uuid4())

    try:
        features = pd.DataFrame([{
            "sepal length (cm)": data.sepal_length,
            "sepal width (cm)": data.sepal_width,
            "petal length (cm)": data.petal_length,
            "petal width (cm)": data.petal_width
        }])

        model = getattr(app.state, "model", None)

        # raise Exception("Test prediction failure")

        if model is None:
            raise HTTPException(
                status_code=500,
                detail="ML model is not available"
            )

        prediction = model.predict(features)[0]

        confidence = None

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features)
            confidence = float(probabilities.max())

        return {
            "prediction": str(prediction),
            "confidence": confidence,
            "model_version": "1.0",
            "request_id": request_id
        }

    except HTTPException:
        raise

    except Exception as error:
        print(f"Prediction error: {error}")
        raise PredictionError()