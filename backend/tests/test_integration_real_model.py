from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings


@pytest.fixture
def real_model_client(client: TestClient) -> TestClient:
    """Fixture that uses the real TFIDFSVMModel if the artifact exists."""
    settings = get_settings()
    if not settings.baseline_model_path.exists():
        pytest.skip(f"Real model not found at {settings.baseline_model_path}")

    # Recreate app without the fake registry
    from app.main import create_app

    app = create_app(settings=settings)
    app.dependency_overrides = client.app.dependency_overrides
    with TestClient(app) as c:
        yield c


def test_predict_real_model(real_model_client: TestClient) -> None:
    r = real_model_client.post(
        "/predict", json={"text": "Pelayanan sangat memuaskan, luar biasa!"}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["label"] == "positive"
    assert 0 <= body["confidence"] <= 1
    assert body["model"] == "tfidf_svm"
