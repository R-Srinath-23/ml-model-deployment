# Iris Flower Classification — REST API

A production-style REST API that serves a trained **Random Forest Classifier** to predict Iris flower species from petal and sepal measurements. Built with **FastAPI** and **scikit-learn**.

---

## 📌 Overview

| | |
|---|---|
| **Problem** | Multi-class classification |
| **Dataset** | Iris (150 samples, 3 classes, balanced) |
| **Model** | Random Forest Classifier |
| **API Framework** | FastAPI + Uvicorn |
| **Serialization** | joblib |

---

## 📊 Dataset

The dataset contains **150 samples** across **3 species** — 50 samples each.

| Feature | Min | Max | Mean |
|---|---|---|---|
| Sepal Length (cm) | 4.3 | 7.9 | 5.84 |
| Sepal Width (cm) | 2.0 | 4.4 | 3.06 |
| Petal Length (cm) | 1.0 | 6.9 | 3.76 |
| Petal Width (cm) | 0.1 | 2.5 | 1.20 |

**Target Classes:**

| ID | Species |
|----|---------|
| 0 | *Iris setosa* |
| 1 | *Iris versicolor* |
| 2 | *Iris virginica* |

---

## 🎯 API Contract

### `POST /predict`

**Request body:**

```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

**Response:**

```json
{
  "predicted_class": "setosa",
  "predicted_class_id": 0,
  "confidence": 0.97
}
```

---

## 🏗️ Project Structure

```
ml-api-project/
├── app/
│   ├── main.py              # FastAPI app entry point
│   ├── models/              # Pydantic request/response schemas
│   └── routers/             # API route definitions
├── ml/
│   ├── train.py             # Model training script
│   └── saved_model/
│       └── model.joblib     # Serialized trained model
├── data/
│   └── iris_dataset.csv     # Dataset
├── tests/                   # Unit and integration tests
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/R-Srinath-23/ml-api-project.git
cd ml-api-project
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🤖 Train the Model

```bash
python ml/train.py
```

This loads `iris_dataset.csv`, trains the Random Forest Classifier, prints accuracy on the test set, and saves the model to `ml/saved_model/model.joblib`.

---

## 🚀 Run the API

```bash
uvicorn app.main:app --reload
```

Server starts at `http://127.0.0.1:8000`

---

## 📖 Interactive Docs

| Interface | URL |
|-----------|-----|
| Swagger UI | http://127.0.0.1:8000/docs |
| ReDoc | http://127.0.0.1:8000/redoc |

---

## 🔌 Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `POST` | `/predict` | Predict Iris species from measurements |

---

**Python:**

```python
import requests

response = requests.post(
    "http://127.0.0.1:8000/predict",
    json={
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
)
print(response.json())
# {"predicted_class": "setosa", "predicted_class_id": 0, "confidence": 0.97}
```

---

## 📦 Dependencies

```
fastapi
uvicorn[standard]
scikit-learn
pandas
joblib
pydantic