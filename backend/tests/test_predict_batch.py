from __future__ import annotations

import csv
import io
from collections.abc import Sequence

from fastapi.testclient import TestClient
from sqlalchemy import select

from app.db import get_db
from app.models import Batch, Prediction
from tests.conftest import TEST_MAX_TEXT_LENGTH, upload


def _make_csv(rows: Sequence[Sequence[str]]) -> str:
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerows(rows)
    return out.getvalue()


def test_batch_upload_success(client: TestClient) -> None:
    content = _make_csv(
        [
            ["text", "date", "other_col"],
            ["Bagus", "2026-01-01", "ignore"],
            ["buruk sekali", "2026-01-02", "me"],
            ["biasa", "", ""],
        ]
    )
    r = upload(client, content, name="My Batch")
    assert r.status_code == 201
    body = r.json()
    assert body["total"] == 3
    assert body["skipped"] == 0
    batch_id = body["batch_id"]

    # Verify DB state directly (get_db is overridden to yield the static pool session)
    db = next(client.app.dependency_overrides[get_db]())  # type: ignore
    batch = db.get(Batch, batch_id)
    assert batch is not None
    assert batch.name == "My Batch"

    preds = db.scalars(
        select(Prediction)
        .where(Prediction.batch_id == batch_id)
        .order_by(Prediction.id)
    ).all()
    assert len(preds) == 3
    assert preds[0].label == "positive"
    assert str(preds[0].source_date) == "2026-01-01"
    assert preds[1].label == "negative"
    assert preds[2].label == "neutral"
    assert preds[2].source_date is None


def test_batch_ignores_empty_lines_and_skips_long(client: TestClient) -> None:
    content = _make_csv(
        [
            ["date", "TEXT"],  # case-insensitive header
            ["", "Bagus"],
            ["2026-01-01", ""],  # empty text -> ignored
            ["", "   "],  # whitespace text -> ignored
            ["", "a" * (TEST_MAX_TEXT_LENGTH + 1)],  # too long -> skipped
        ]
    )
    r = upload(client, content)
    assert r.status_code == 201
    assert r.json()["total"] == 1
    assert r.json()["skipped"] == 1


def test_batch_all_lines_empty_or_skipped(client: TestClient) -> None:
    content = _make_csv([["text"], [""], ["a" * 1000]])
    r = upload(client, content)
    assert r.status_code == 422
    assert "no valid rows" in r.json()["detail"].lower()


def test_batch_missing_text_column(client: TestClient) -> None:
    content = _make_csv([["date", "comment"], ["2026-01-01", "Bagus"]])
    r = upload(client, content)
    assert r.status_code == 422
    assert "must contain a 'text' column" in r.json()["detail"].lower()


def test_batch_invalid_date_rejects_whole_file(client: TestClient) -> None:
    content = _make_csv(
        [
            ["text", "date"],
            ["Bagus", "2026-01-01"],
            ["Jelek", "01/02/2026"],  # invalid format
        ]
    )
    r = upload(client, content)
    assert r.status_code == 422
    assert "invalid date '01/02/2026' on line 3" in r.json()["detail"].lower()


def test_batch_invalid_csv_format(client: TestClient) -> None:
    # Not UTF-8
    r = upload(client, b"\xff\xfeT\x00e\x00x\x00t\x00")
    assert r.status_code == 400
    assert "not valid utf-8" in r.json()["detail"].lower()


def test_batch_file_too_large(client: TestClient) -> None:
    # TEST_MAX_UPLOAD_MB is 0.01 (~10KB)
    content = "text\n" + ("a\n" * 20000)
    r = upload(client, content)
    assert r.status_code == 413
    assert "too large" in r.json()["detail"].lower()


def test_batch_unknown_model(client: TestClient) -> None:
    content = _make_csv([["text"], ["Bagus"]])
    r = upload(client, content, model="gpt")
    assert r.status_code == 400
    assert "unknown model 'gpt'" in r.json()["detail"].lower()
