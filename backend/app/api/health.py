from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.api.deps import get_registry
from app.db import get_db
from app.schemas import HealthResponse
from app.services.model_registry import ModelRegistry

router = APIRouter(tags=["health"])
logger = logging.getLogger(__name__)


@router.get("/health", response_model=HealthResponse)
def health(
    db: Annotated[Session, Depends(get_db)],
    registry: Annotated[ModelRegistry, Depends(get_registry)],
) -> HealthResponse:
    """Always 200; ``status`` is ``degraded`` if the DB or the default model is not usable."""
    try:
        db.execute(text("SELECT 1"))
        database = "ok"
    except SQLAlchemyError as e:
        logger.error("Database health check failed: %s", e)
        database = f"error: {e.__class__.__name__}"
    models = registry.status()
    healthy = database == "ok" and models.get(registry.default_model) == "loaded"
    return HealthResponse(
        status="ok" if healthy else "degraded",
        database=database,
        models=models,
        default_model=registry.default_model,
    )
