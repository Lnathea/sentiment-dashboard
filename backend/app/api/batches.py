from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import Batch, Prediction
from app.schemas import BatchStats, BatchSummary, ErrorResponse
from app.services.stats import compute_batch_stats

router = APIRouter(prefix="/batches", tags=["batches"])


@router.get("", response_model=list[BatchSummary])
def list_batches(db: Annotated[Session, Depends(get_db)]) -> list[BatchSummary]:
    """All batches, newest first, with the number of stored predictions."""
    total = func.count(Prediction.id).label("total")
    rows = db.execute(
        select(Batch, total)
        .outerjoin(Prediction, Prediction.batch_id == Batch.id)
        .group_by(Batch.id)
        .order_by(Batch.created_at.desc(), Batch.id.desc())
    ).all()
    return [
        BatchSummary(
            id=b.id, name=b.name, source=b.source, created_at=b.created_at, total=int(n)
        )
        for b, n in rows
    ]


@router.get(
    "/{batch_id}/stats",
    response_model=BatchStats,
    responses={404: {"model": ErrorResponse}},
)
def batch_stats(batch_id: int, db: Annotated[Session, Depends(get_db)]) -> BatchStats:
    batch = db.get(Batch, batch_id)
    if batch is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Batch {batch_id} not found.")
    return BatchStats.model_validate(compute_batch_stats(db, batch))
