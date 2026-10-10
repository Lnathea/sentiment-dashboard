"""Shared test fixtures.

- Database: a fresh SQLite **in-memory** database per test (StaticPool so every
  session sees the same connection), injected by overriding ``get_db``.
- Model: a deterministic ``FakeModel`` registered in a test ``ModelRegistry``,
  so endpoint tests do not need the trained .joblib file.
- Settings: explicit values (small limits) injected by overriding ``get_settings``.
"""

from __future__ import annotations

import os

# Must be set before `app` is imported so the module-level engine never touches
# a real database file.
os.environ["DATABASE_URL"] = "sqlite://"

from collections.abc import Iterator, Sequence

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.config import Settings, get_settings
from app.db import Base, get_db, make_engine
from app.main import create_app
from app.services.model_registry import ModelRegistry

TEST_MAX_TEXT_LENGTH = 50
TEST_MAX_UPLOAD_MB = 0.01  # ~10 KB


class FakeModel:
    """Keyword rules: 'bagus' -> positive, 'buruk' -> negative, else neutral."""

    name = "fake"

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def predict(self, texts: Sequence[str]) -> list[tuple[str, float]]:
        self.calls.append(list(texts))
        out = []
        for t in texts:
            low = t.lower()
            if "bagus" in low:
                out.append(("positive", 0.9))
            elif "buruk" in low:
                out.append(("negative", 0.8))
            else:
                out.append(("neutral", 0.6))
        return out


@pytest.fixture
def settings() -> Settings:
    return Settings(
        _env_file=None,
        database_url="sqlite://",
        max_text_length=TEST_MAX_TEXT_LENGTH,
        max_upload_mb=TEST_MAX_UPLOAD_MB,
        default_model="fake",
        cors_origins="http://localhost:3000",
    )


@pytest.fixture
def session_factory() -> Iterator[sessionmaker[Session]]:
    engine = make_engine("sqlite://", poolclass=StaticPool)
    Base.metadata.create_all(engine)
    yield sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    engine.dispose()


@pytest.fixture
def fake_model() -> FakeModel:
    return FakeModel()


@pytest.fixture
def registry(fake_model: FakeModel) -> ModelRegistry:
    reg = ModelRegistry(default_model="fake")
    reg.register(fake_model)
    reg.register_failure("broken", "model file not found at '/nowhere/model.joblib'")
    return reg


@pytest.fixture
def client(
    settings: Settings, registry: ModelRegistry, session_factory: sessionmaker[Session]
) -> Iterator[TestClient]:
    app = create_app(settings=settings, registry=registry)

    def _get_db() -> Iterator[Session]:
        db = session_factory()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = _get_db
    app.dependency_overrides[get_settings] = lambda: settings
    with TestClient(app) as c:  # context manager runs the lifespan
        yield c


def upload(
    client: TestClient,
    content: str | bytes,
    filename: str = "data.csv",
    **form: str,
):
    """POST a CSV to /predict/batch."""
    data = content.encode("utf-8") if isinstance(content, str) else content
    return client.post(
        "/predict/batch",
        files={"file": (filename, data, "text/csv")},
        data=form,
    )
