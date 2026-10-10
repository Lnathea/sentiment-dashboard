"""Dashboard statistics for one batch.

Counts and averages are computed with portable SQL aggregates (COUNT/AVG +
GROUP BY). The per-day trend and top words are computed in Python, because
truncating a timestamp to a date is not portable (SQLite's CAST(x AS DATE)
returns a number, PostgreSQL uses ::date / date_trunc).
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from datetime import date, datetime

from ml.preprocess import preprocess
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import SENTIMENT_LABELS, Batch, Prediction

TOP_WORDS_PER_LABEL = 20

# Small Indonesian stopword list (common function words + frequent informal spellings).
STOPWORDS: frozenset[str] = frozenset(
    [
        "ada",
        "adalah",
        "agar",
        "akan",
        "aku",
        "al",
        "anda",
        "apa",
        "atau",
        "bagi",
        "bahwa",
        "banyak",
        "baru",
        "begitu",
        "belum",
        "bisa",
        "buat",
        "dan",
        "dari",
        "dengan",
        "di",
        "dia",
        "dong",
        "ga",
        "gak",
        "gk",
        "itu",
        "ini",
        "jadi",
        "jika",
        "juga",
        "kalau",
        "kami",
        "kamu",
        "karena",
        "kau",
        "ke",
        "kita",
        "lagi",
        "lah",
        "mau",
        "me",
        "mereka",
        "nya",
        "nih",
        "oleh",
        "pada",
        "para",
        "pun",
        "saja",
        "sama",
        "saya",
        "se",
        "sangat",
        "sebagai",
        "sedang",
        "sehingga",
        "sekali",
        "selalu",
        "semua",
        "sih",
        "saat",
        "sudah",
        "tak",
        "tapi",
        "telah",
        "tentang",
        "tersebut",
        "tetapi",
        "tidak",
        "udah",
        "untuk",
        "ya",
        "yaitu",
        "yang",
        "yg",
        "dgn",
        "utk",
        "tdk",
        "sdh",
        "aja",
        "kok",
        "deh",
        "kan",
        "pas",
        "punya",
        "hanya",
        "lebih",
        "masih",
    ]
)

_TOKEN_RE = re.compile(r"[a-z]+(?:-[a-z]+)*")  # keeps reduplication like "baik-baik"


def tokenize(text: str) -> list[str]:
    """Shared preprocess, then letters-only tokens, minus stopwords and 1-char tokens."""
    return [
        t
        for t in _TOKEN_RE.findall(preprocess(text))
        if len(t) > 1 and t not in STOPWORDS
    ]


def _as_date(value: date | datetime) -> date:
    return value.date() if isinstance(value, datetime) else value


def compute_batch_stats(db: Session, batch: Batch) -> dict:
    # --- counts and average confidence (SQL aggregates) ---
    agg_rows = db.execute(
        select(
            Prediction.label, func.count(Prediction.id), func.avg(Prediction.confidence)
        )
        .where(Prediction.batch_id == batch.id)
        .group_by(Prediction.label)
    ).all()
    counts = {label: 0 for label in SENTIMENT_LABELS}
    avg_conf: dict[str, float | None] = {label: None for label in SENTIMENT_LABELS}
    for label, n, avg in agg_rows:
        counts[label] = int(n)
        avg_conf[label] = round(float(avg), 4) if avg is not None else None
    total = sum(counts.values())
    percentages = {
        label: (round(100 * counts[label] / total, 2) if total else 0.0)
        for label in SENTIMENT_LABELS
    }

    # --- per-day trend and top words (Python) ---
    per_day: dict[date, Counter[str]] = defaultdict(Counter)
    words: dict[str, Counter[str]] = {label: Counter() for label in SENTIMENT_LABELS}
    rows = db.execute(
        select(
            Prediction.label,
            Prediction.text,
            Prediction.source_date,
            Prediction.created_at,
        ).where(Prediction.batch_id == batch.id)
    )
    for label, text, source_date, created_at in rows:
        day = source_date if source_date is not None else _as_date(created_at)
        per_day[day][label] += 1
        words[label].update(tokenize(text))

    trend = [
        {
            "date": day,
            **{label: per_day[day][label] for label in SENTIMENT_LABELS},
            "total": sum(per_day[day].values()),
        }
        for day in sorted(per_day)
    ]
    top_words = {
        label: [
            {"word": w, "count": c}
            # sort by count desc, then word asc for deterministic ties
            for w, c in sorted(words[label].items(), key=lambda kv: (-kv[1], kv[0]))[
                :TOP_WORDS_PER_LABEL
            ]
        ]
        for label in SENTIMENT_LABELS
    }

    return {
        "batch_id": batch.id,
        "name": batch.name,
        "total": total,
        "counts": counts,
        "percentages": percentages,
        "avg_confidence": avg_conf,
        "trend": trend,
        "top_words": top_words,
    }
