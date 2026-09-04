from fastapi import APIRouter, HTTPException, Request

import pandas as pd

from app.config import settings
from app.models.schemas import (
    PredictionInput,
    PredictionV2Output,
)

from app.exceptions import PredictionError
from app.logging_config import logger


router = APIRouter(
    prefix="/api/v2"
)


@router.post(
    "/predict",
    response_model=PredictionV2Output
)
def predict_v2(
    data: PredictionInput,
    request: Request
):

    request_id = request.state.request_id

    try:
        model = getattr(
            request.app.state,
            "model",
            None
        )

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

        if not hasattr(model, "predict_proba"):
            raise PredictionError()

        probabilities = model.predict_proba(features)[0]

        class_probabilities = {
            str(class_name): float(probability)
            for class_name, probability in zip(
                model.classes_,
                probabilities
            )
        }

        logger.info(
            f"prediction_v2_success "
            f"request_id={request_id} "
            f"prediction={prediction[0]} "
            f"probabilities={class_probabilities}"
        )

        return PredictionV2Output(
            prediction=str(prediction[0]),
            probabilities=class_probabilities,
            model_version=settings.MODEL_VERSION,
            request_id=request_id
        )

    except HTTPException:
        raise

    except Exception as error:
        logger.error(
            f"prediction_v2_failed "
            f"request_id={request_id} "
            f"error={error}"
        )

        raise PredictionError()