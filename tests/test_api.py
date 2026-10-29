import json
from pathlib import Path

from fastapi.testclient import TestClient

from main import app

client = TestClient(app)
sample = json.loads((Path(__file__).parent / "sample_student.json").read_text())


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ui():
    response = client.get("/")
    assert response.status_code == 200
    assert "Student Academic Risk" in response.text


def test_predict():
    response = client.post("/predict", json=sample)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in {"Dropout", "Enrolled", "Graduate"}
    assert abs(sum(body["probabilities"].values()) - 1) < 0.01
