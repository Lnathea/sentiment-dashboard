from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SentimentLabel = Literal["positive", "neutral", "negative"]


# ---------- /predict ----------
class PredictRequest(BaseModel):
    # Emptiness and MAX_TEXT_LENGTH are checked in the endpoint (they depend on settings).
    text: str = Field(
        ...,
        description="Text to classify (Indonesian).",
        examples=["Pelayanannya ramah sekali"],
    )
    model: str | None = Field(
        None,
        description="Model name; defaults to DEFAULT_MODEL.",
        examples=["tfidf_svm"],
    )


class PredictResponse(BaseModel):
    label: SentimentLabel
    confidence: float = Field(..., ge=0, le=1)
    model: str


class BatchUploadResponse(BaseModel):
    batch_id: int
    total: int = Field(..., description="Number of rows predicted and stored.")
    skipped: int = Field(
        ..., description="Rows skipped because text exceeded MAX_TEXT_LENGTH."
    )


# ---------- /batches ----------
class BatchSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    source: str
    created_at: datetime
    total: int


class LabelCounts(BaseModel):
    positive: int
    neutral: int
    negative: int


class LabelFloats(BaseModel):
    positive: float | None
    neutral: float | None
    negative: float | None


class TrendPoint(BaseModel):
    date: date
    positive: int
    neutral: int
    negative: int
    total: int


class WordCount(BaseModel):
    word: str
    count: int


class TopWords(BaseModel):
    positive: list[WordCount]
    neutral: list[WordCount]
    negative: list[WordCount]


class BatchStats(BaseModel):
    batch_id: int
    name: str
    total: int
    counts: LabelCounts
    percentages: LabelFloats = Field(
        ..., description="Percent of total (0-100) per label."
    )
    avg_confidence: LabelFloats = Field(
        ..., description="Mean confidence per label; null if no rows."
    )
    trend: list[TrendPoint] = Field(
        ...,
        description="Per-day counts. Uses source_date if present, else the UTC date of created_at.",
    )
    top_words: TopWords


# ---------- /health ----------
class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    database: str
    models: dict[str, str]
    default_model: str


class ErrorResponse(BaseModel):
    detail: str
