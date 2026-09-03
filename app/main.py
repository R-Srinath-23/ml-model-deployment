from contextlib import asynccontextmanager

from pathlib import Path
import uuid
import time

import joblib

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.config import settings
from app.exceptions import PredictionError

from app.logging_config import logger


from app.routers.v1 import router as v1_router

# MODEL_PATH = Path(__file__).resolve().parent.parent / "ml" / "saved_model" / "model.joblib"

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading ML model...")
    model_path = Path(settings.MODEL_PATH)

    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )
    
    app.state.model = joblib.load(model_path)
    logger.info("ML model loaded successfully.")

    
    yield

    logger.info("Shutting down API...")


app = FastAPI(
    title=settings.API_TITLE,
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

app.include_router(v1_router)