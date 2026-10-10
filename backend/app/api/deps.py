"""Shared FastAPI dependencies (overridable in tests)."""

from __future__ import annotations

from fastapi import Request

from app.services.model_registry import ModelRegistry


def get_registry(request: Request) -> ModelRegistry:
    return request.app.state.registry
