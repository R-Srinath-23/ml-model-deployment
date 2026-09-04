# Iris Flower Classification — Production-Style REST API

A production-style Machine Learning REST API that serves a trained **Random Forest Classifier** to predict Iris flower species from sepal and petal measurements.

The API is built using **FastAPI** and **scikit-learn** and includes model loading at startup, request validation, confidence scores, batch prediction, structured logging, environment-based configuration, API versioning, error handling, model metadata, and automated tests.

---

## 📌 Overview

| Component               | Technology                                |
| ----------------------- | ----------------------------------------- |
| **Problem**             | Multi-class classification                |
| **Dataset**             | Iris Dataset                              |
| **Samples**             | 150                                       |
| **Classes**             | 3                                         |
| **Model**               | Random Forest Classifier                  |
| **API Framework**       | FastAPI                                   |
| **ASGI Server**         | Uvicorn                                   |
| **Data Validation**     | Pydantic                                  |
| **Model Serialization** | joblib                                    |
| **Data Processing**     | Pandas                                    |
| **Testing**             | Pytest + FastAPI TestClient               |
| **Configuration**       | Environment variables + Pydantic Settings |
| **Logging**             | Python logging + RotatingFileHandler      |
| **API Versioning**      | `/api/v1` and `/api/v2`                   |

---

# 📊 Dataset

The project uses the classic **Iris dataset** containing:

* 150 samples
* 4 input features
* 3 target classes
* 50 samples per class

### Features

| Feature      | Unit |
| ------------ | ---- |
| Sepal Length | cm   |
| Sepal Width  | cm   |
| Petal Length | cm   |
| Petal Width  | cm   |

### Target Classes

| Class | Species         |
| ----- | --------------- |
| 0     | Iris setosa     |
| 1     | Iris versicolor |
| 2     | Iris virginica  |

The trained model exposes the class names during prediction:

```text
setosa
versicolor
virginica
```

---

# 🤖 Machine Learning Model

The project uses a **Random Forest Classifier** from scikit-learn.

The trained model is serialized using `joblib` and stored at:

```text
ml/saved_model/model.joblib
```

The FastAPI application loads the model **once during application startup** rather than loading it for every prediction request.

This improves API efficiency and avoids unnecessary model loading overhead.

---

# 🏗️ Current Project Structure

```text
ml_api_project/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   ├── exceptions.py
│   ├── logging_config.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py
│   │
│   └── routers/
│       ├── __init__.py
│       ├── v1.py
│       └── v2.py
│
├── ml/
│   └── saved_model/
│       └── model.joblib
│
├── logs/
│   └── api.log
│
├── tests/
│   ├── conftest.py
│   ├── test_health.py
│   ├── test_predict.py
│   ├── test_batch.py
│   ├── test_model_info.py
│   └── test_v2.py
│
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

---

# 🔄 Development Tasks Completed

The project was developed incrementally through **14 tasks**.

## Task 1 — Plan the ML API Architecture

Designed the overall architecture for serving a trained Iris classification model through a REST API.

The basic architecture consists of:

```text
Client
   │
   ▼
FastAPI REST API
   │
   ▼
Request Validation
   │
   ▼
Random Forest Model
   │
   ▼
Prediction Response
```

---

## Task 2 — Project Setup

Created the initial project structure and Python virtual environment.

The main components include:

```text
app/
ml/
tests/
requirements.txt
.gitignore
```

A dedicated virtual environment is used to isolate project dependencies.

---

## Task 3 — Train and Save the ML Model

A Random Forest Classifier was trained using the Iris dataset.

The trained model was serialized using `joblib`:

```text
ml/saved_model/model.joblib
```

The saved model can then be loaded by the FastAPI application.

---

## Task 4 — Create the Basic FastAPI Application

Created the initial FastAPI application and configured Uvicorn as the development server.

The application can be started with:

```bash
uvicorn app.main:app --reload
```

The API documentation is automatically available through Swagger UI.

---

# 🚀 API Development

## Task 5 — Load the Model at Application Startup

The ML model is loaded using FastAPI's `lifespan` mechanism.

Instead of loading the model for every request, it is loaded once:

```text
Application Startup
       │
       ▼
Load model.joblib
       │
       ▼
Store in app.state.model
       │
       ▼
Handle API requests
```

The model is accessed using:

```python
request.app.state.model
```

If the model file does not exist, application startup fails with an appropriate error.

---

## Task 6 — Add Pydantic Input Validation

Created a Pydantic request schema:

```python
class PredictionInput(BaseModel):

    sepal_length: float = Field(..., gt=0)
    sepal_width: float = Field(..., gt=0)
    petal_length: float = Field(..., gt=0)
    petal_width: float = Field(..., gt=0)
```

This ensures:

* Required fields are present
* Input values are numeric
* Measurements must be greater than zero

Invalid requests return HTTP `422`.

---

## Task 7 — Build the Core Prediction API

The prediction endpoint was implemented using:

```text
POST /api/v1/predict
```

Example request:

```json
{
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
}
```

Example response:

```json
{
    "prediction": "setosa",
    "confidence": 1.0,
    "model_version": "1.0",
    "request_id": "..."
}
```

The confidence value is calculated using:

```python
model.predict_proba()
```

---

# Task 8 — Response Models and Error Handling

Created Pydantic response models to ensure consistent API responses.

The v1 response model contains:

```text
prediction
confidence
model_version
request_id
```

Custom prediction errors are handled using a `PredictionError` exception.

The API returns:

```json
{
    "detail": "Prediction failed"
}
```

for prediction failures.

---

# Task 9 — Structured Request Logging

Implemented structured request logging.

Each request receives a unique:

```text
request_id
```

The API logs information such as:

```text
request_id
HTTP method
request path
status code
request duration
prediction result
prediction confidence
errors
```

Example:

```text
request_id=...
method=POST
path=/api/v1/predict
status_code=200
duration=0.0142s
```

Logs are stored in:

```text
logs/api.log
```

---

# Task 10 — API Versioning

Introduced API versioning using FastAPI routers.

The API structure became:

```text
/api/v1/
```

The current v1 endpoints are:

```text
GET  /api/v1/health
POST /api/v1/predict
POST /api/v1/predict-batch
GET  /api/v1/model-info
```

API versioning allows future breaking changes to be introduced without immediately breaking existing clients.

---

# Task 11 — Multiple API Endpoints

Added additional endpoints to make the API more practical.

### Health Endpoint

```text
GET /api/v1/health
```

Example response:

```json
{
    "status": "ok",
    "model_loaded": true
}
```

### Batch Prediction

```text
POST /api/v1/predict-batch
```

Allows multiple Iris measurements to be predicted in a single API request.

### Model Information

```text
GET /api/v1/model-info
```

Example response:

```json
{
    "model_loaded": true,
    "model_type": "RandomForestClassifier",
    "model_version": "1.0",
    "training_date": "2026-08-23",
    "expected_features": [
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)"
    ]
}
```

---

# Task 12 — Configuration Management

Moved configurable values away from hardcoded application logic.

Configuration is managed using environment variables and Pydantic Settings.

Example `.env`:

```env
MODEL_PATH=ml/saved_model/model.joblib
LOG_LEVEL=INFO
MAX_BATCH_SIZE=100
API_TITLE=Iris ML Prediction API
MODEL_VERSION=1.0
MODEL_TYPE=RandomForestClassifier
TRAINING_DATE=2026-08-23
```

The batch prediction limit is controlled by:

```env
MAX_BATCH_SIZE=100
```

This allows configuration to be changed without modifying the application source code.

---

# Task 13 — Automated API Testing

Implemented automated tests using:

* Pytest
* FastAPI TestClient

Tests cover:

### Health

```text
GET /api/v1/health
```

### Prediction

```text
POST /api/v1/predict
```

Tests include:

* Valid input
* Missing fields
* Invalid field types
* Negative values

### Batch Prediction

```text
POST /api/v1/predict-batch
```

Tests include:

* Valid batch
* Batch exceeding `MAX_BATCH_SIZE`

### Model Information

```text
GET /api/v1/model-info
```

The tests verify that the model metadata is returned correctly.

---

# Task 14 — API v2 and Breaking Change

Introduced a new API version:

```text
/api/v2/predict
```

The v1 API remains unchanged.

## v1 Response

```json
{
    "prediction": "setosa",
    "confidence": 1.0,
    "model_version": "1.0",
    "request_id": "..."
}
```

## v2 Response

The v2 API replaces the single `confidence` value with the complete probability distribution:

```json
{
    "prediction": "setosa",
    "probabilities": {
        "setosa": 1.0,
        "versicolor": 0.0,
        "virginica": 0.0
    },
    "model_version": "1.0",
    "request_id": "..."
}
```

This is an intentional **breaking response-shape change**.

Existing clients can continue using:

```text
/api/v1/predict
```

while new clients can use:

```text
/api/v2/predict
```

---

# 🧪 API Version Testing

The project contains automated tests verifying that both API versions work simultaneously.

The test sends the same input to:

```text
/api/v1/predict
/api/v2/predict
```

and verifies that their response schemas are different.

### v1

```text
prediction
confidence
model_version
request_id
```

### v2

```text
prediction
probabilities
model_version
request_id
```

The test also verifies that:

```text
confidence
```

exists in v1 but not v2, while:

```text
probabilities
```

exists in v2 but not v1.

---

# 📡 API Endpoints

## API v1

| Method | Endpoint                | Description                   |
| ------ | ----------------------- | ----------------------------- |
| `GET`  | `/api/v1/health`        | Check API and model health    |
| `POST` | `/api/v1/predict`       | Predict a single Iris species |
| `POST` | `/api/v1/predict-batch` | Predict multiple Iris samples |
| `GET`  | `/api/v1/model-info`    | Return model metadata         |

## API v2

| Method | Endpoint          | Description                                      |
| ------ | ----------------- | ------------------------------------------------ |
| `POST` | `/api/v2/predict` | Predict with full class probability distribution |

---

# 📥 Prediction Request

Both v1 and v2 use the same input structure:

```json
{
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
}
```

---

# 📤 v1 Prediction Response

```json
{
    "prediction": "setosa",
    "confidence": 1.0,
    "model_version": "1.0",
    "request_id": "..."
}
```

---

# 📤 v2 Prediction Response

```json
{
    "prediction": "setosa",
    "probabilities": {
        "setosa": 1.0,
        "versicolor": 0.0,
        "virginica": 0.0
    },
    "model_version": "1.0",
    "request_id": "..."
}
```

---

# 📦 Batch Prediction

Endpoint:

```text
POST /api/v1/predict-batch
```

Example request:

```json
{
    "inputs": [
        {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        },
        {
            "sepal_length": 6.2,
            "sepal_width": 3.4,
            "petal_length": 5.4,
            "petal_width": 2.3
        }
    ]
}
```

The maximum batch size is configured through:

```env
MAX_BATCH_SIZE=100
```

Requests exceeding the configured limit return HTTP `422`.

---

# ⚙️ Configuration

Application configuration is managed through:

```text
app/config.py
```

Example configuration:

```text
MODEL_PATH
LOG_LEVEL
MAX_BATCH_SIZE
API_TITLE
MODEL_VERSION
MODEL_TYPE
TRAINING_DATE
```

Environment variables are loaded from:

```text
.env
```

---

# 📝 Logging

Application logs are written to:

```text
logs/api.log
```

The logging system records:

* Application startup
* Model loading
* Application shutdown
* Request IDs
* HTTP methods
* Request paths
* Response status codes
* Request duration
* Prediction success
* Prediction failures
* Batch size violations

---

# 🧪 Running Tests

Activate the virtual environment:

### Windows

```powershell
.\mlvevn\Scripts\activate
```

Run all tests:

```bash
python -m pytest -v
```

Current test suite:

```text
10 passed
```

The tests cover the v1 API, batch prediction, health endpoint, model information, validation, and v2 API behavior.

---

# 🚀 Setup and Installation

## 1. Clone the repository

```bash
git clone https://github.com/R-Srinath-23/ml-api-project.git
cd ml-api-project
```

## 2. Create a virtual environment

```bash
python -m venv mlvevn
```

### Windows

```powershell
.\mlvevn\Scripts\activate
```

### macOS/Linux

```bash
source mlvevn/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

# 🚀 Run the API

Start the FastAPI application using:

```bash
uvicorn app.main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

---

# 📖 Interactive API Documentation

FastAPI automatically provides interactive documentation.

### Swagger UI

```text
http://127.0.0.1:8000/docs
```

### ReDoc

```text
http://127.0.0.1:8000/redoc
```

Swagger can be used to test:

```text
/api/v1/health
/api/v1/predict
/api/v1/predict-batch
/api/v1/model-info
/api/v2/predict
```

---

# 🐍 Python Client Example

## v1

```python
import requests

response = requests.post(
    "http://127.0.0.1:8000/api/v1/predict",
    json={
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
)

print(response.json())
```

Example:

```json
{
    "prediction": "setosa",
    "confidence": 1.0,
    "model_version": "1.0",
    "request_id": "..."
}
```

## v2

```python
import requests

response = requests.post(
    "http://127.0.0.1:8000/api/v2/predict",
    json={
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
)

print(response.json())
```

Example:

```json
{
    "prediction": "setosa",
    "probabilities": {
        "setosa": 1.0,
        "versicolor": 0.0,
        "virginica": 0.0
    },
    "model_version": "1.0",
    "request_id": "..."
}
```

---

# 📦 Core Dependencies

The project uses the following core Python packages:

```text
fastapi
uvicorn
scikit-learn
pandas
joblib
pydantic
pydantic-settings
pytest
```

---

# 🔐 Environment Configuration

Sensitive or environment-specific configuration should not be committed to Git.

The `.env` file should be included in `.gitignore` when appropriate.

Example:

```text
.env
logs/
__pycache__/
.pytest_cache/
mlvevn/
```

---

# 🎯 Key Features

The completed project currently supports:

* ✅ Random Forest ML model
* ✅ FastAPI REST API
* ✅ Model loading at application startup
* ✅ Pydantic request validation
* ✅ Structured API responses
* ✅ Confidence scores
* ✅ Full class probability distribution in v2
* ✅ Health monitoring
* ✅ Batch prediction
* ✅ Configurable batch size
* ✅ Model metadata endpoint
* ✅ Environment-based configuration
* ✅ Request IDs
* ✅ Request duration logging
* ✅ Prediction error handling
* ✅ API versioning
* ✅ v1/v2 compatibility
* ✅ Automated Pytest tests
* ✅ 10 passing tests

---

# 🧠 Key Engineering Concepts Demonstrated

This project demonstrates practical experience with:

* REST API development
* FastAPI
* Machine Learning model serving
* Model serialization
* Pydantic validation
* API response contracts
* HTTP status codes
* Exception handling
* Structured logging
* Configuration management
* API versioning
* Batch processing
* Automated testing
* Regression testing
* Backward compatibility
* Breaking API changes

---

# 🔮 Future Improvements

Potential future improvements include:

* Docker containerization
* CI/CD pipeline
* API authentication
* Rate limiting
* Prometheus metrics
* Model performance monitoring
* Model version management
* Database integration
* Cloud deployment
* Automated model retraining
* Production deployment with multiple workers

---

# 👨‍💻 Project Status

**Tasks 1–14 completed successfully.**

Latest test result:

```text
10 passed, 1 warning
```

The warning is related to the current Starlette/httpx TestClient dependency compatibility and does not cause test failures.

The next development stage is **Docker containerization** of the FastAPI ML API.