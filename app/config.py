from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    MODEL_PATH: str = "ml/saved_model/model.joblib"
    LOG_LEVEL: str = "INFO"
    MAX_BATCH_SIZE: int = 100
    API_TITLE: str = "Iris ML Prediction API"

    MODEL_VERSION: str = "1.0"
    MODEL_TYPE: str = "RandomForestClassifier"
    TRAINING_DATE: str = "2026-08-23"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True
    )


settings = Settings()