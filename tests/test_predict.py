def test_predict_valid_input(client):
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }

    response = client.post(
        "/api/v1/predict",
        json=payload
    )

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data
    assert "confidence" in data
    assert "model_version" in data
    assert "request_id" in data

    assert isinstance(data["prediction"], str)
    assert data["confidence"] is not None
    assert 0 <= data["confidence"] <= 1


def test_predict_missing_field(client):
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4
    }

    response = client.post(
        "/api/v1/predict",
        json=payload
    )

    assert response.status_code == 422


def test_predict_invalid_field_type(client):
    payload = {
        "sepal_length": "hello",
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }

    response = client.post(
        "/api/v1/predict",
        json=payload
    )

    assert response.status_code == 422


def test_predict_invalid_negative_value(client):
    payload = {
        "sepal_length": -5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }

    response = client.post(
        "/api/v1/predict",
        json=payload
    )

    assert response.status_code == 422