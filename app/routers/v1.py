from fastapi import APIRouter, HTTPException, Request

import pandas as pd

from app.models.schemas import PredictionInput, PredictionOutput
from app.exceptions import PredictionError
from app.logging_config import logger


router = APIRouter(
    prefix="/api/v1"
)


@router.get("/health")
def health(request: Request):
    model_loaded = getattr(request.app.state, "model", None) is not None

    return {
        "status": "ok",
        "model_loaded": model_loaded
    }


@router.post("/predict", response_model=PredictionOutput)
def predict(data: PredictionInput, request: Request):

    request_id = request.state.request_id

    try:

        features = pd.DataFrame([{
            "sepal length (cm)": data.sepal_length,
            "sepal width (cm)": data.sepal_width,
            "petal length (cm)": data.petal_length,
            "petal width (cm)": data.petal_width
        }])

        model = getattr(request.app.state, "model", None)

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