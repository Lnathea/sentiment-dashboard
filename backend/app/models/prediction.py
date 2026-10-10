from __future__ import annotations

from datetime import date, datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    CheckConstraint,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base
from app.models.batch import utcnow

if TYPE_CHECKING:
    from app.models.batch import Batch

# Canonical label set used by the API and the database (lowercase English).
SENTIMENT_LABELS: tuple[str, ...] = ("positive", "neutral", "negative")


class Prediction(Base):
    __tablename__ = "predictions"
    __table_args__ = (
        # Portable alternative to a DB-specific ENUM type.
        CheckConstraint(
            "label IN ('positive', 'neutral', 'negative')", name="ck_predictions_label"
        ),
        CheckConstraint(
            "confidence >= 0 AND confidence <= 1", name="ck_predictions_confidence"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    batch_id: Mapped[int] = mapped_column(
        ForeignKey("batches.id", ondelete="CASCADE"), nullable=False, index=True
    )
    text: Mapped[str] = mapped_column(Text, nullable=False)
    label: Mapped[str] = mapped_column(String(16), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    model: Mapped[str] = mapped_column(String(50), nullable=False)
    source_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow
    )

    batch: Mapped[Batch] = relationship(back_populates="predictions")
