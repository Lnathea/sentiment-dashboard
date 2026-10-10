"""Sentiment model interface and name-based registry.

Endpoints only talk to :class:`ModelRegistry`. Adding a new model (e.g. IndoBERT
in Phase 2) means: implement :class:`SentimentModel`, then add a loader to
``MODEL_LOADERS`` in :func:`build_registry`. No endpoint changes are needed.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Protocol, runtime_checkable

from app.config import Settings

logger = logging.getLogger(__name__)

Prediction = tuple[str, float]  # (label, confidence in [0, 1])


@runtime_checkable
class SentimentModel(Protocol):
    """Anything with a ``name`` and a batch ``predict`` method."""

    name: str

    def predict(self, texts: Sequence[str]) -> list[Prediction]:
        """Return one ``(label, confidence)`` per input text, same order."""
        ...


class ModelLoadError(Exception):
    """A model could not be loaded (missing file, bad format, ...)."""


class UnknownModelError(LookupError):
    """The requested model name is not registered at all."""

    def __init__(self, name: str, available: Sequence[str]) -> None:
        self.name = name
        self.available = list(available)
        super().__init__(
            f"Unknown model '{name}'. Available models: {', '.join(self.available) or 'none'}."
        )


class ModelUnavailableError(RuntimeError):
    """The model name is known but the model failed to load."""

    def __init__(self, name: str, reason: str) -> None:
        self.name = name
        self.reason = reason
        super().__init__(f"Model '{name}' is not available: {reason}")


class ModelRegistry:
    def __init__(self, default_model: str) -> None:
        self.default_model = default_model
        self._models: dict[str, SentimentModel] = {}
        self._failures: dict[str, str] = {}

    def register(self, model: SentimentModel) -> None:
        self._models[model.name] = model
        self._failures.pop(model.name, None)

    def register_failure(self, name: str, reason: str) -> None:
        self._models.pop(name, None)
        self._failures[name] = reason

    def names(self) -> list[str]:
        return sorted({*self._models, *self._failures})

    def get(self, name: str | None = None) -> SentimentModel:
        name = name or self.default_model
        if name in self._models:
            return self._models[name]
        if name in self._failures:
            raise ModelUnavailableError(name, self._failures[name])
        raise UnknownModelError(name, self.names())

    def status(self) -> dict[str, str]:
        out = {name: "loaded" for name in self._models}
        out.update(
            {name: f"unavailable: {reason}" for name, reason in self._failures.items()}
        )
        return dict(sorted(out.items()))


ModelLoader = Callable[[Settings], SentimentModel]


def _load_tfidf_svm(settings: Settings) -> SentimentModel:
    from app.services.tfidf_svm import TFIDFSVMModel

    return TFIDFSVMModel.load(settings.baseline_model_path)


# name -> loader. Phase 2: add "indobert": _load_indobert here.
MODEL_LOADERS: dict[str, ModelLoader] = {
    "tfidf_svm": _load_tfidf_svm,
}


def build_registry(settings: Settings) -> ModelRegistry:
    """Load every known model. Failures are recorded, never raised, so the API
    can still start and report a clear error (HTTP 503 + /health)."""
    registry = ModelRegistry(default_model=settings.default_model)
    for name, loader in MODEL_LOADERS.items():
        try:
            registry.register(loader(settings))
            logger.info("Model '%s' loaded", name)
        except ModelLoadError as e:
            registry.register_failure(name, str(e))
            logger.error("Model '%s' could not be loaded: %s", name, e)
    return registry
