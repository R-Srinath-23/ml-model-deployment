def test_predict_batch_valid(client):
    payload = {
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

    response = client.post(
        "/api/v1/predict-batch",
        json=payload
    )

    assert response.status_code == 200

    data = response.json()

    assert "predictions" in data
    assert len(data["predictions"]) == 2


def test_predict_batch_oversized(client):
    payload = {
        "inputs": [
            {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            }
            for _ in range(101)
        ]
    }

    response = client.post(
        "/api/v1/predict-batch",
        json=payload
    )

    assert response.status_code == 422

    data = response.json()

    assert "detail" in data
    assert "100" in data["detail"]