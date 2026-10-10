from __future__ import annotations

import csv
import io

from fastapi.testclient import TestClient

from tests.conftest import upload


def _seed_batch(client: TestClient, name: str, rows: list[tuple[str, str]]) -> int:
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(["text", "date"])
    writer.writerows(rows)
    r = upload(client, out.getvalue(), name=name)
    assert r.status_code == 201
    return r.json()["batch_id"]


def test_list_batches(client: TestClient) -> None:
    b1 = _seed_batch(client, "Batch A", [("bagus", "")])
    b2 = _seed_batch(client, "Batch B", [("buruk sekali", ""), ("buruk", "")])

    r = client.get("/batches")
    assert r.status_code == 200
    body = r.json()
    assert len(body) == 2
    # newest first
    assert body[0]["id"] == b2
    assert body[0]["name"] == "Batch B"
    assert body[0]["total"] == 2
    assert body[1]["id"] == b1
    assert body[1]["name"] == "Batch A"
    assert body[1]["total"] == 1


def test_batch_stats(client: TestClient) -> None:
    rows = [
        ("Kamera ini bagus", "2026-01-01"),
        ("Bagus dan murah", "2026-01-01"),
        ("Pelayanan buruk sekali", "2026-01-02"),
        ("Biasa saja", "2026-01-02"),
    ]
    b_id = _seed_batch(client, "Test Stats", rows)

    r = client.get(f"/batches/{b_id}/stats")
    assert r.status_code == 200
    stats = r.json()

    assert stats["name"] == "Test Stats"
    assert stats["total"] == 4

    # fake model: 2 positive, 1 negative, 1 neutral
    assert stats["counts"] == {"positive": 2, "neutral": 1, "negative": 1}
    assert stats["percentages"] == {"positive": 50.0, "neutral": 25.0, "negative": 25.0}

    # Trend (using source_date)
    trend = stats["trend"]
    assert len(trend) == 2
    assert trend[0]["date"] == "2026-01-01"
    assert trend[0]["positive"] == 2
    assert trend[1]["date"] == "2026-01-02"
    assert trend[1]["negative"] == 1
    assert trend[1]["neutral"] == 1

    # Top words
    # "kamera" (1), "bagus" (2), "murah" (1), "pelayanan" (1), "buruk" (1), "sekali" (stopword, so 0)
    top_pos = stats["top_words"]["positive"]
    assert {"word": "bagus", "count": 2} in top_pos
    assert {"word": "kamera", "count": 1} in top_pos
    assert {"word": "murah", "count": 1} in top_pos

    top_neg = stats["top_words"]["negative"]
    # equal counts are ordered alphabetically (deterministic tie-break)
    assert top_neg[0] == {"word": "buruk", "count": 1}
    assert top_neg[1] == {"word": "pelayanan", "count": 1}
    assert not any(w["word"] == "sekali" for w in top_neg)  # "sekali" is in stopwords


def test_batch_stats_not_found(client: TestClient) -> None:
    r = client.get("/batches/999/stats")
    assert r.status_code == 404
