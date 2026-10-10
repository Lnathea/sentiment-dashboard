from __future__ import annotations

from fastapi.testclient import TestClient

from tests.conftest import TEST_MAX_TEXT_LENGTH, FakeModel


def test_predict_default_model(client: TestClient) -> None:
    r = client.post("/predict", json={"text": "Filmnya bagus sekali"})
    assert r.status_code == 200
    assert r.json() == {"label": "positive", "confidence": 0.9, "model": "fake"}


def test_predict_explicit_model(client: TestClient) -> None:
    r = client.post("/predict", json={"text": "pelayanan buruk", "model": "fake"})
    assert r.status_code == 200
    assert r.json()["label"] == "negative"


def test_predict_strips_text_before_model(
    client: TestClient, fake_model: FakeModel
) -> None:
    client.post("/predict", json={"text": "   biasa saja  "})
    assert fake_model.calls[-1] == ["biasa saja"]


def test_predict_empty_text(client: TestClient) -> None:
    for text in ["", "   ", "\n\t"]:
        r = client.post("/predict", json={"text": text})
        assert r.status_code == 422
        assert "must not be empty" in r.json()["detail"]


def test_predict_text_too_long(client: TestClient) -> None:
    r = client.post("/predict", json={"text": "a" * (TEST_MAX_TEXT_LENGTH + 1)})
    assert r.status_code == 422
    assert "too long" in r.json()["detail"]


def test_predict_text_at_limit_ok(client: TestClient) -> None:
    r = client.post("/predict", json={"text": "a" * TEST_MAX_TEXT_LENGTH})
    assert r.status_code == 200


def test_predict_missing_field(client: TestClient) -> None:
    r = client.post("/predict", json={})
    assert r.status_code == 422
    body = r.json()
    assert isinstance(body["detail"], str)
    assert "text" in body["detail"]


def test_predict_wrong_type(client: TestClient) -> None:
    r = client.post("/predict", json={"text": 123})
    assert r.status_code == 422


def test_predict_unknown_model(client: TestClient) -> None:
    r = client.post("/predict", json={"text": "halo", "model": "gpt"})
    assert r.status_code == 400
    assert "Unknown model 'gpt'" in r.json()["detail"]
    assert "fake" in r.json()["detail"]


def test_predict_model_unavailable(client: TestClient) -> None:
    r = client.post("/predict", json={"text": "halo", "model": "broken"})
    assert r.status_code == 503
    assert "not available" in r.json()["detail"]
    assert "model file not found" in r.json()["detail"]
