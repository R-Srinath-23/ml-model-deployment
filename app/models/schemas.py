from typing import List
from pydantic import BaseModel, Field


class PredictionInput(BaseModel):
    sepal_length: float = Field(..., gt=0)
    sepal_width: float = Field(..., gt=0)
    petal_length: float = Field(..., gt=0)
    petal_width: float = Field(..., gt=0)

class PredictionOutput(BaseModel):
    prediction: str
    confidence: float | None
    model_version: str
    request_id: str

class PredictionBatchInput(BaseModel):
    inputs: List[PredictionInput] = Field(..., min_length=1, max_length=100)


class PredictionBatchOutput(BaseModel):
    predictions: List[PredictionOutput]