"""FastAPI application entry point: ``uvicorn app.main:app --reload`` (from backend/)."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import batches, health, predict
from app.config import Settings, get_settings
from app.services.model_registry import (
    ModelRegistry,
    ModelUnavailableError,
    UnknownModelError,
    build_registry,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")


def _format_validation_errors(exc: RequestValidationError) -> str:
    parts = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err.get("loc", ()) if p != "body")
        parts.append(f"{loc}: {err.get('msg')}" if loc else str(err.get("msg")))
    return "Invalid request. " + "; ".join(parts)


def create_app(
    settings: Settings | None = None, registry: ModelRegistry | None = None
) -> FastAPI:
    """Build the app. Tests pass a ready ``registry`` so no model file is loaded."""
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.registry = (
            registry if registry is not None else build_registry(settings)
        )
        yield

    app = FastAPI(
        title="Sentiment Dashboard API",
        description="Indonesian sentiment analysis (positive / neutral / negative).",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.exception_handler(UnknownModelError)
    async def _unknown_model(_: Request, exc: UnknownModelError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST, content={"detail": str(exc)}
        )

    @app.exception_handler(ModelUnavailableError)
    async def _model_unavailable(
        _: Request, exc: ModelUnavailableError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": str(exc)},
        )

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        # Same {"detail": "<message>"} shape as every other error, plus the raw list.
        return JSONResponse(
            status_code=422,
            content={
                "detail": _format_validation_errors(exc),
                "errors": jsonable_encoder(exc.errors()),
            },
        )

    app.include_router(health.router)
    app.include_router(predict.router)
    app.include_router(batches.router)
    return app


app = create_app()
