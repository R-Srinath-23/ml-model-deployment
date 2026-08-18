# ML Model Deployment as a Monitored REST API

## Project Overview

This project will deploy a machine learning classification model as a REST API using FastAPI. The API will accept input data, validate it, pass it to a trained machine learning model, and return the prediction as a JSON response.

## Dataset

The project uses the Iris dataset provided by Scikit-learn through `load_iris()`.

The dataset contains four input features:

- Sepal length
- Sepal width
- Petal length
- Petal width

The model will classify the flower into one of three species:

- Setosa
- Versicolor
- Virginica

## Machine Learning Problem

This is a supervised classification problem.

The goal is to predict the species of an Iris flower based on its sepal and petal measurements.

## Machine Learning Model

The planned model is:

**RandomForestClassifier**

The model will be trained using the Iris dataset and saved so that it can later be loaded by the FastAPI application.

## API Contract

The `/predict` endpoint accepts four numerical values representing the sepal length, sepal width, petal length, and petal width of an Iris flower. The API validates the input and sends the validated values to the trained RandomForestClassifier model. The model predicts the Iris flower species as Setosa, Versicolor, or Virginica. The API returns the predicted species as a JSON response.

### Example Request

```json
{
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
}