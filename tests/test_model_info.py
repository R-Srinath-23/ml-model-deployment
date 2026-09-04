def test_model_info(client):
    response = client.get("/api/v1/model-info")

    assert response.status_code == 200

    data = response.json()

    assert "model_loaded" in data
    assert "model_type" in data
    assert "model_version" in data
    assert "training_date" in data
    assert "expected_features" in data

    assert data["model_loaded"] is True
    assert data["model_type"] == "RandomForestClassifier"
    assert data["model_version"] == "1.0"