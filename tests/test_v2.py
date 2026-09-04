def test_predict_v2_valid_input(client):

    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }

    response = client.post(
        "/api/v2/predict",
        json=payload
    )

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data
    assert "probabilities" in data
    assert "model_version" in data
    assert "request_id" in data

    assert "confidence" not in data

    assert isinstance(data["prediction"], str)

    assert isinstance(data["probabilities"], dict)

    assert set(data["probabilities"].keys()) == {
    "setosa",
    "versicolor",
    "virginica"
}
    assert abs(
        sum(data["probabilities"].values()) - 1.0
    ) < 0.000001

def test_v1_and_v2_response_shapes(client):

    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }

    v1_response = client.post(
        "/api/v1/predict",
        json=payload
    )

    v2_response = client.post(
        "/api/v2/predict",
        json=payload
    )

    assert v1_response.status_code == 200
    assert v2_response.status_code == 200

    v1_data = v1_response.json()
    v2_data = v2_response.json()

    # Check v1 response shape
    assert set(v1_data.keys()) == {
        "prediction",
        "confidence",
        "model_version",
        "request_id"
    }

    # Check v2 response shape
    assert set(v2_data.keys()) == {
        "prediction",
        "probabilities",
        "model_version",
        "request_id"
    }

    # Prove the breaking change
    assert "confidence" in v1_data
    assert "confidence" not in v2_data

    assert "probabilities" not in v1_data
    assert "probabilities" in v2_data

    # Check v1 result
    assert isinstance(
        v1_data["prediction"],
        str
    )

    assert 0 <= v1_data["confidence"] <= 1

    # Check v2 result
    assert isinstance(
        v2_data["prediction"],
        str
    )

    assert isinstance(
        v2_data["probabilities"],
        dict
    )

    assert set(
        v2_data["probabilities"].keys()
    ) == {
        "setosa",
        "versicolor",
        "virginica"
    }

    assert abs(
        sum(v2_data["probabilities"].values()) - 1.0
    ) < 0.000001