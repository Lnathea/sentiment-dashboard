"""TF-IDF + linear classifier baseline (trained by ml/train_baseline.py)."""

from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from ml.preprocess import preprocess_many

from app.models.prediction import SENTIMENT_LABELS
from app.services.model_registry import ModelLoadError, Prediction

logger = logging.getLogger(__name__)


class TFIDFSVMModel:
    name = "tfidf_svm"

    def __init__(self, pipeline: Any, meta: dict[str, Any] | None = None) -> None:
        if not hasattr(pipeline, "predict_proba") or not hasattr(pipeline, "classes_"):
            raise ModelLoadError(
                "model object has no predict_proba/classes_ (not a fitted classifier)"
            )
        unknown = set(map(str, pipeline.classes_)) - set(SENTIMENT_LABELS)
        if unknown:
            raise ModelLoadError(f"model has unexpected labels: {sorted(unknown)}")
        self._pipeline = pipeline
        self._classes = np.asarray(pipeline.classes_).astype(str)
        self.meta = meta or {}

    @classmethod
    def load(cls, path: Path) -> TFIDFSVMModel:
        if not path.is_file():
            raise ModelLoadError(
                f"model file not found at '{path}'. Train it with `python ml/train_baseline.py` "
                "or set BASELINE_MODEL_PATH."
            )
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                obj = joblib.load(path)
        except Exception as e:  # corrupt file, incompatible pickle, ...
            raise ModelLoadError(f"could not read model file '{path}': {e}") from e
        for w in caught:
            logger.warning("While loading %s: %s", path.name, w.message)

        if not isinstance(obj, dict) or "pipeline" not in obj:
            raise ModelLoadError(
                f"'{path}' is not a model artifact produced by train_baseline.py"
            )
        meta = obj.get("meta", {})
        trained_with = meta.get("sklearn_version")
        if trained_with and trained_with != sklearn.__version__:
            logger.warning(
                "Model trained with scikit-learn %s but running %s; predictions may differ.",
                trained_with,
                sklearn.__version__,
            )
        return cls(obj["pipeline"], meta)

    def predict(self, texts: Sequence[str]) -> list[Prediction]:
        if not texts:
            return []
        proba = self._pipeline.predict_proba(preprocess_many(texts))
        best = proba.argmax(axis=1)
        return [
            (str(self._classes[i]), float(proba[row, i])) for row, i in enumerate(best)
        ]
