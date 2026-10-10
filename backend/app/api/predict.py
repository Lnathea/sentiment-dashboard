from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session

from app.api.deps import get_registry
from app.config import Settings, get_settings
from app.db import get_db
from app.models import Batch, Prediction
from app.schemas import (
    BatchUploadResponse,
    ErrorResponse,
    PredictRequest,
    PredictResponse,
)
from app.services.csv_parser import (
    CSVContentError,
    CSVFormatError,
    ParsedRow,
    parse_csv,
)
from app.services.model_registry import ModelRegistry, SentimentModel

router = APIRouter(tags=["predict"])

PREDICT_CHUNK_SIZE = 1000
READ_CHUNK_BYTES = 1024 * 1024
ERRORS = {
    400: {"model": ErrorResponse},
    413: {"model": ErrorResponse},
    422: {"model": ErrorResponse},
    503: {"model": ErrorResponse},
}


@router.post("/predict", response_model=PredictResponse, responses=ERRORS)
def predict(
    body: PredictRequest,
    settings: Annotated[Settings, Depends(get_settings)],
    registry: Annotated[ModelRegistry, Depends(get_registry)],
) -> PredictResponse:
    """Classify one text. Not stored in the database."""
    text = body.text.strip()
    if not text:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, "Field 'text' must not be empty."
        )
    if len(text) > settings.max_text_length:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            f"Field 'text' is too long ({len(text)} characters); "
            f"maximum is {settings.max_text_length}.",
        )
    model = registry.get(body.model)
    label, confidence = model.predict([text])[0]
    return PredictResponse(label=label, confidence=confidence, model=model.name)


async def _read_limited(file: UploadFile, max_bytes: int) -> bytes:
    """Read the upload but stop as soon as it exceeds the limit."""
    buf = bytearray()
    while chunk := await file.read(READ_CHUNK_BYTES):
        buf.extend(chunk)
        if len(buf) > max_bytes:
            raise HTTPException(
                status.HTTP_413_CONTENT_TOO_LARGE,
                f"File is too large; maximum size is {max_bytes / (1024 * 1024):g} MB.",
            )
    return bytes(buf)


@router.post(
    "/predict/batch",
    response_model=BatchUploadResponse,
    status_code=status.HTTP_201_CREATED,
    responses=ERRORS,
)
async def predict_batch(
    file: Annotated[
        UploadFile,
        File(
            description="UTF-8 CSV with a 'text' column; optional 'date' (YYYY-MM-DD)."
        ),
    ],
    settings: Annotated[Settings, Depends(get_settings)],
    registry: Annotated[ModelRegistry, Depends(get_registry)],
    db: Annotated[Session, Depends(get_db)],
    model: Annotated[
        str | None, Form(description="Model name; defaults to DEFAULT_MODEL.")
    ] = None,
    name: Annotated[
        str | None,
        Form(max_length=255, description="Batch name; defaults to the file name."),
    ] = None,
) -> BatchUploadResponse:
    """Upload a CSV, classify every row and store the results as one batch."""
    # Resolve the model first so a bad model name fails before reading the file.
    sentiment_model = registry.get(model)

    content = await _read_limited(file, settings.max_upload_bytes)
    try:
        parsed = parse_csv(content, settings.max_text_length)
    except CSVFormatError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e)) from e
    except CSVContentError as e:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(e)) from e

    if not parsed.rows:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "CSV contains no valid rows to analyze "
            f"({parsed.ignored_empty} empty, {parsed.skipped_too_long} longer than "
            f"{settings.max_text_length} characters).",
        )

    batch_name = (name or "").strip() or (file.filename or "upload.csv")[:255]
    # Model inference + DB writes are blocking: run them off the event loop.
    batch_id = await run_in_threadpool(
        _store_batch, db, sentiment_model, batch_name, parsed.rows
    )
    return BatchUploadResponse(
        batch_id=batch_id, total=len(parsed.rows), skipped=parsed.skipped_too_long
    )


def _store_batch(
    db: Session, sentiment_model: SentimentModel, batch_name: str, rows: list[ParsedRow]
) -> int:
    batch = Batch(name=batch_name, source="csv_upload")
    try:
        db.add(batch)
        db.flush()  # assigns batch.id
        for start in range(0, len(rows), PREDICT_CHUNK_SIZE):
            chunk = rows[start : start + PREDICT_CHUNK_SIZE]
            results = sentiment_model.predict([r.text for r in chunk])
            db.add_all(
                Prediction(
                    batch_id=batch.id,
                    text=row.text,
                    label=label,
                    confidence=confidence,
                    model=sentiment_model.name,
                    source_date=row.source_date,
                )
                for row, (label, confidence) in zip(chunk, results, strict=True)
            )
        db.commit()
    except Exception:
        db.rollback()
        raise
    return batch.id
