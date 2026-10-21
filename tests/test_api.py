from fastapi.testclient import TestClient
from apps.api.main import app, runner

client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["app"] == "AMILCAR AI"


def test_chat_reports_untrained_model_cleanly(monkeypatch) -> None:
    monkeypatch.setattr(runner, "available", False)
    response = client.post("/api/chat", json={"message": "Hola"})
    assert response.status_code == 503
    assert "todavía no está entrenado" in response.json()["detail"]
