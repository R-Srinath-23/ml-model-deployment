from contextlib import asynccontextmanager

from pathlib import Path
import uuid
import time

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from app.logging_config import logger
from app.models.schemas import PredictionInput, PredictionOutput

MODEL_PATH = Path(__file__).resolve().parent.parent / "ml" / "saved_model" / "model.joblib"

class PredictionError(Exception):
    pass

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading ML model...")

    app.state.model = joblib.load(MODEL_PATH)

    logger.info("ML model loaded successfully.")


    yield

    logger.info("Shutting down API...")


app = FastAPI(
    title="Iris ML Prediction API",
    lifespan=lifespan
)

@app.middleware("http")
async def log_requests(request: Request, call_next):

    request_id = str(uuid.uuid4())
    request.state.request_id = request_id

    start_time = time.perf_counter()

    try:
        response = await call_next(request)

    except Exception as error:
        duration = time.perf_counter() - start_time

        logger.error(
            f"request_id={request_id} "
            f"method={request.method} "
            f"path={request.url.path} "
            f"duration={duration:.4f}s "
            f"error={error}"
        )

        raise

    duration = time.perf_counter() - start_time

    logger.info(
        f"request_id={request_id} "
        f"method={request.method} "
        f"path={request.url.path} "
        f"status_code={response.status_code} "
        f"duration={duration:.4f}s"
    )

    response.headers["X-Request-ID"] = request_id

    return response


@app.exception_handler(PredictionError)
async def prediction_error_handler(request: Request, exc):
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
def predict(data: PredictionInput, request: Request):

    request_id = request.state.request_id

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

        logger.info(
            f"prediction_success "
            f"request_id={request_id} "
            f"prediction={prediction} "
            f"confidence={confidence}"
        )


        return {
            "prediction": str(prediction),
            "confidence": confidence,
            "model_version": "1.0",
            "request_id": request_id
        }

    except HTTPException:
        raise

    except Exception as error:
        logger.error(
            f"prediction_failed "
            f"request_id={request_id} "
            f"error={error}"
        )

        raise PredictionError()