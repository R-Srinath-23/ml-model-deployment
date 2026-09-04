def test_health(client):
    response = client.get("/api/v1/health")

    assert response.status_code == 200

    data = response.json()

    assert "status" in data
    assert "model_loaded" in data

    assert data["status"] == "ok"
    assert data["model_loaded"] is True