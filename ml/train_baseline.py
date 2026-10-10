"""Train and evaluate the TF-IDF + linear classifier baseline on SmSA.

Protocol (no test leakage):
1. Use the official IndoNLU splits (train / valid / test) as-is.
2. Every text goes through ``ml.preprocess.preprocess`` (same function the
   backend imports at inference time). It is called explicitly and is NOT
   pickled inside the pipeline, so the .joblib file only contains sklearn objects.
3. TF-IDF is fitted on the TRAIN split only (it lives inside the Pipeline).
4. Candidates: CalibratedClassifierCV(LinearSVC) and LogisticRegression, each with
   C in {0.1, 1, 10}. Model selection uses macro-F1 on VALID only.
5. The selected model (fitted on train) is evaluated ONCE on TEST.

Usage (from the repo root, after ``python ml/download_data.py``):
    python ml/train_baseline.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")  # headless: write PNG without a display
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

ML_DIR = Path(__file__).resolve().parent
REPO_ROOT = ML_DIR.parent
if str(REPO_ROOT) not in sys.path:
    # Import preprocess exactly like the backend does: ``ml.preprocess``.
    sys.path.insert(0, str(REPO_ROOT))

from ml.preprocess import preprocess_many

DATA_DIR = ML_DIR / "data" / "smsa"
ARTIFACTS_DIR = ML_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "tfidf_svm.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
CM_PATH = ARTIFACTS_DIR / "confusion_matrix.png"

LABELS = ["positive", "neutral", "negative"]
RANDOM_STATE = 42
C_GRID = [0.1, 1.0, 10.0]


def load_split(name: str) -> tuple[list[str], list[str]]:
    path = DATA_DIR / f"{name}.tsv"
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run `python ml/download_data.py` first.")
    texts: list[str] = []
    labels: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        text, label = line.rsplit("\t", 1)
        texts.append(text)
        labels.append(label.strip())
    return texts, labels


def make_tfidf() -> TfidfVectorizer:
    return TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        lowercase=False,  # already lowercased by preprocess()
    )


def make_candidates() -> list[tuple[str, dict[str, Any], Pipeline]]:
    candidates = []
    for c in C_GRID:
        svm = CalibratedClassifierCV(
            LinearSVC(C=c, random_state=RANDOM_STATE),
            method="sigmoid",
            cv=5,
        )
        candidates.append(
            (
                "linear_svc_calibrated",
                {"C": c},
                Pipeline([("tfidf", make_tfidf()), ("clf", svm)]),
            )
        )
    for c in C_GRID:
        lr = LogisticRegression(C=c, max_iter=5000, random_state=RANDOM_STATE)
        candidates.append(
            (
                "logistic_regression",
                {"C": c},
                Pipeline([("tfidf", make_tfidf()), ("clf", lr)]),
            )
        )
    return candidates


def scores(y_true: list[str], y_pred: list[str]) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def plot_confusion_matrix(cm: np.ndarray, path: Path, title: str) -> None:
    row_sums = cm.sum(axis=1, keepdims=True)
    pct = np.divide(
        cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0
    )
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(pct, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, label="fraction of true class")
    ax.set_xticks(range(len(LABELS)), LABELS)
    ax.set_yticks(range(len(LABELS)), LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            color = "white" if pct[i, j] > 0.5 else "black"
            ax.text(
                j,
                i,
                f"{cm[i, j]}\n({pct[i, j]:.0%})",
                ha="center",
                va="center",
                color=color,
            )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    t0 = time.time()
    x_train_raw, y_train = load_split("train")
    x_valid_raw, y_valid = load_split("valid")
    x_test_raw, y_test = load_split("test")
    print(f"Rows: train={len(y_train)} valid={len(y_valid)} test={len(y_test)}")

    x_train = preprocess_many(x_train_raw)
    x_valid = preprocess_many(x_valid_raw)

    # ---- model selection on VALID ----
    results: list[dict[str, Any]] = []
    fitted: list[Pipeline] = []
    for name, params, pipe in make_candidates():
        start = time.time()
        pipe.fit(x_train, y_train)
        y_pred = pipe.predict(x_valid).tolist()
        proba = pipe.predict_proba(x_valid)
        row = {
            "classifier": name,
            "params": params,
            **scores(y_valid, y_pred),
            "log_loss": float(log_loss(y_valid, proba, labels=pipe.classes_)),
            "fit_seconds": round(time.time() - start, 2),
        }
        results.append(row)
        fitted.append(pipe)
        print(
            f"  valid | {name:<22} C={params['C']:<5} "
            f"acc={row['accuracy']:.4f} macroF1={row['macro_f1']:.4f} logloss={row['log_loss']:.4f}"
        )

    # Highest macro-F1; ties broken by accuracy, then lower log-loss.
    best_idx = max(
        range(len(results)),
        key=lambda i: (
            results[i]["macro_f1"],
            results[i]["accuracy"],
            -results[i]["log_loss"],
        ),
    )
    best = results[best_idx]
    best_pipe = fitted[best_idx]

    def best_of(clf_name: str) -> dict[str, Any]:
        rows = [r for r in results if r["classifier"] == clf_name]
        return max(rows, key=lambda r: (r["macro_f1"], r["accuracy"], -r["log_loss"]))

    best_svm = best_of("linear_svc_calibrated")
    best_lr = best_of("logistic_regression")
    reason = (
        f"Selected {best['classifier']} (C={best['params']['C']}) because it has the highest "
        f"macro-F1 on the validation split ({best['macro_f1']:.4f}). Best per family on valid: "
        f"LinearSVC+calibration C={best_svm['params']['C']} macro-F1={best_svm['macro_f1']:.4f}; "
        f"LogisticRegression C={best_lr['params']['C']} macro-F1={best_lr['macro_f1']:.4f}. "
        "Macro-F1 is used because the classes are imbalanced."
    )
    print(reason)

    # ---- final evaluation on TEST (used exactly once) ----
    x_test = preprocess_many(x_test_raw)
    y_test_pred = best_pipe.predict(x_test).tolist()
    test_scores = scores(y_test, y_test_pred)
    report = classification_report(
        y_test, y_test_pred, labels=LABELS, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_test, y_test_pred, labels=LABELS)
    print(
        f"TEST  | acc={test_scores['accuracy']:.4f} macroF1={test_scores['macro_f1']:.4f}"
    )
    print(
        classification_report(
            y_test, y_test_pred, labels=LABELS, digits=4, zero_division=0
        )
    )
    print("Confusion matrix (rows=true, cols=pred):", LABELS)
    print(cm)

    # Informational only (not used for any decision): exact overlap train<->test.
    train_set = set(x_train)
    overlap_pre = sum(1 for t in x_test if t in train_set)
    overlap_raw = len(set(x_test_raw) & set(x_train_raw))

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    trained_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    meta = {
        "model_name": "tfidf_svm",
        "classifier": best["classifier"],
        "params": best["params"],
        "labels": list(best_pipe.classes_),
        "sklearn_version": sklearn.__version__,
        "trained_at": trained_at,
        "preprocess": "ml.preprocess.preprocess (applied outside the pipeline)",
        "random_state": RANDOM_STATE,
    }
    joblib.dump(
        {"pipeline": best_pipe, "labels": list(best_pipe.classes_), "meta": meta},
        MODEL_PATH,
        compress=3,
    )

    metrics = {
        "dataset": "SmSA (IndoNLU), official splits",
        "split_sizes": {
            "train": len(y_train),
            "valid": len(y_valid),
            "test": len(y_test),
        },
        "label_order": LABELS,
        "tfidf": {"ngram_range": [1, 2], "min_df": 2, "sublinear_tf": True},
        "validation": {
            "candidates": results,
            "selected_index": best_idx,
            "reason": reason,
        },
        "test": {
            **test_scores,
            "per_class": {
                lbl: {k: float(v) for k, v in report[lbl].items()} for lbl in LABELS
            },
            "confusion_matrix": {"labels": LABELS, "matrix": cm.tolist()},
        },
        "data_notes": {
            "test_texts_also_in_train_after_preprocess": overlap_pre,
            "test_texts_also_in_train_raw_unique": overlap_raw,
        },
        "model": meta,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    title = (
        f"SmSA test - {best['classifier']} C={best['params']['C']}\n"
        f"acc={test_scores['accuracy']:.3f}  macro-F1={test_scores['macro_f1']:.3f}"
    )
    plot_confusion_matrix(cm, CM_PATH, title)
    print(
        f"Test texts also present in train (after preprocess): {overlap_pre}/{len(x_test)}"
    )
    print(f"Saved: {MODEL_PATH}\n       {METRICS_PATH}\n       {CM_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
