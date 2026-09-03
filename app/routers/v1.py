from fastapi import APIRouter, HTTPException, Request

import pandas as pd

from app.config import settings
from app.models.schemas import (
    PredictionInput,
    PredictionOutput,
    PredictionBatchInput,
    PredictionBatchOutput,
)

from app.exceptions import PredictionError
from app.logging_config import logger


router = APIRouter(
    prefix="/api/v1"
)

# Model metadata
# MODEL_VERSION = "1.0"
# MODEL_TYPE = "RandomForestClassifier"
# TRAINING_DATE = "2026-08-23"

# EXPECTED_FEATURES = [
#     "sepal length (cm)",
#     "sepal width (cm)",
#     "petal length (cm)",
#     "petal width (cm)",
# ]


@router.get("/health")
def health(request: Request):
    model = getattr(request.app.state, "model", None)

    return {
        "status": "ok",
        "model_loaded": model is not None
    }


@router.post("/predict", response_model=PredictionOutput)
def predict(data: PredictionInput, request: Request):

    request_id = request.state.request_id

    try:
        model = getattr(request.app.state, "model", None)

        if model is None:
            raise HTTPException(
                status_code=500,
                detail="ML model is not available"
            )

        features = pd.DataFrame([{
            "sepal length (cm)": data.sepal_length,
            "sepal width (cm)": data.sepal_width,
            "petal length (cm)": data.petal_length,
            "petal width (cm)": data.petal_width
        }])

        prediction = model.predict(features)

        confidence = None

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features)
            confidence = float(probabilities.max())

        logger.info(
            f"prediction_success "
            f"request_id={request_id} "
            f"prediction={prediction[0]} "
            f"confidence={confidence}"
        )

        return PredictionOutput(
            prediction=str(prediction[0]),
            confidence=confidence,
            model_version=settings.MODEL_VERSION,
            request_id=request_id
        )


    except HTTPException:
        raise

    except Exception as error:
        logger.error(
            f"prediction_failed "
            f"request_id={request_id} "
            f"error={error}"
        )

        raise PredictionError()

@router.post(
    "/predict-batch",
    response_model=PredictionBatchOutput
)
def predict_batch(
    data: PredictionBatchInput,
    request: Request
):

    request_id = request.state.request_id
    batch_size = len(data.inputs)

     # Enforce configuration value
    if batch_size > settings.MAX_BATCH_SIZE:

        logger.warning(
            f"batch_size_exceeded "
            f"request_id={request_id} "
            f"batch_size={batch_size} "
            f"max_batch_size={settings.MAX_BATCH_SIZE}"
        )

        raise HTTPException(
            status_code=422,
            detail=(
                f"Batch size cannot exceed "
                f"{settings.MAX_BATCH_SIZE}"
            )
        )

    try:
        model = getattr(request.app.state, "model", None)

        if model is None:
            raise HTTPException(
                status_code=500,
                detail="ML model is not available"
            )

        # Convert all input data into one DataFrame
        features = pd.DataFrame([
            {
                "sepal length (cm)": item.sepal_length,
                "sepal width (cm)": item.sepal_width,
                "petal length (cm)": item.petal_length,
                "petal width (cm)": item.petal_width
            }
            for item in data.inputs
        ])

        # Make predictions for the complete batch
        predictions = model.predict(features)

        # Calculate confidence for each prediction
        confidences = [None] * batch_size

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features)

            confidences = probabilities.max(axis=1).tolist()

        # Create response for each prediction
        results = []

        for prediction, confidence in zip(
            predictions,
            confidences
        ):
            results.append(
                PredictionOutput(
                    prediction=str(prediction),
                    confidence=(
                        float(confidence)
                        if confidence is not None
                        else None
                    ),
                    model_version=settings.MODEL_VERSION,
                    request_id=request_id
                )
            )

        logger.info(
            f"batch_prediction_success "
            f"request_id={request_id} "
            f"batch_size={batch_size}"
        )

        return PredictionBatchOutput(
            predictions=results
        )

    except HTTPException:
        raise

    except Exception as error:
        logger.error(
            f"batch_prediction_failed "
            f"request_id={request_id} "
            f"batch_size={batch_size} "
            f"error={error}"
        )

        raise PredictionError()


# --------------------------------------------------
# Model Information
# --------------------------------------------------

@router.get("/model-info")
def model_info(request: Request):

    model = getattr(request.app.state, "model", None)

    return {
        "model_loaded": model is not None,
        "model_type": settings.MODEL_TYPE,
        "model_version": settings.MODEL_VERSION,
        "training_date": settings.TRAINING_DATE,
        "expected_features": [
            "sepal length (cm)",
            "sepal width (cm)",
            "petal length (cm)",
            "petal width (cm)"
        ]
    }