from __future__ import annotations

from fastapi.testclient import TestClient

from app.services.model_registry import ModelRegistry


def test_health_ok(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["database"] == "ok"
    assert body["default_model"] == "fake"
    assert body["models"]["fake"] == "loaded"
    assert body["models"]["broken"].startswith("unavailable: model file not found")


def test_health_degraded_when_default_model_missing(
    client: TestClient, registry: ModelRegistry
) -> None:
    registry.default_model = "broken"
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "degraded"


def test_cors_allows_configured_origin(client: TestClient) -> None:
    r = client.options(
        "/predict",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert r.headers.get("access-control-allow-origin") == "http://localhost:3000"


def test_cors_rejects_other_origin(client: TestClient) -> None:
    r = client.get("/health", headers={"Origin": "http://evil.example"})
    assert "access-control-allow-origin" not in r.headers
