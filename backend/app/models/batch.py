from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base

if TYPE_CHECKING:
    from app.models.prediction import Prediction


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Batch(Base):
    __tablename__ = "batches"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    source: Mapped[str] = mapped_column(String(50), nullable=False)
    # Python-side default (not a DB server default) keeps behaviour identical
    # on SQLite and PostgreSQL.
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow
    )

    predictions: Mapped[list[Prediction]] = relationship(
        back_populates="batch", cascade="all, delete-orphan", passive_deletes=True
    )
