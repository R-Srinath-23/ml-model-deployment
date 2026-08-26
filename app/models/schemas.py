from pydantic import BaseModel, Field


class PredictionInput(BaseModel):
    sepal_length: float = Field(
        ...,
        gt=0,
        le=10,
        description="Sepal length must be between 0 and 10"
    )

    sepal_width: float = Field(
        ...,
        gt=0,
        description="Sepal width must be greater than 0"
    )

    petal_length: float = Field(
        ...,
        gt=0,
        description="Petal length must be greater than 0"
    )

    petal_width: float = Field(
        ...,
        gt=0,
        description="Petal width must be greater than 0"
    )

class PredictionOutput(BaseModel):
    prediction: str
    confidence: float | None
    model_version: str
    request_id: str