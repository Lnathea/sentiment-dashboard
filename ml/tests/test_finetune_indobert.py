"""Tests for the pure helpers in ml/finetune_indobert.py.

No model is downloaded and nothing is trained. Tests that need torch are skipped
automatically when torch is not installed.
"""

from __future__ import annotations

import argparse

import numpy as np
import pytest

from ml import finetune_indobert as ft


def _fake_tokenizer(texts: list[str], truncation: bool, max_length: int) -> dict:
    """Stands in for a HF tokenizer: one token id per character, truncated."""
    assert truncation is True
    ids = [[ord(c) for c in t][:max_length] for t in texts]
    return {"input_ids": ids, "attention_mask": [[1] * len(x) for x in ids]}


# ---------------------------- label mapping ----------------------------
def test_label_mapping_is_the_contract() -> None:
    assert ft.ID2LABEL == {0: "negative", 1: "neutral", 2: "positive"}
    assert ft.LABEL2ID == {"negative": 0, "neutral": 1, "positive": 2}


def test_encode_decode_roundtrip() -> None:
    labels = ["positive", "negative", "neutral", "positive"]
    ids = ft.encode_labels(labels)
    assert ids == [2, 0, 1, 2]
    assert ft.decode_labels(ids) == labels


def test_encode_labels_rejects_unknown_label() -> None:
    with pytest.raises(ValueError, match="unexpected label"):
        ft.encode_labels(["positive", "happy"])


# ------------------------------- metrics -------------------------------
def test_compute_metrics_perfect_and_imperfect() -> None:
    labels = np.array([0, 1, 2, 2])
    perfect = np.eye(3)[labels]
    m = ft.compute_metrics((perfect, labels))
    assert m == {"accuracy": 1.0, "macro_f1": 1.0}

    # always predicts class 2: accuracy 2/4; F1 = 0, 0, 2/3 -> macro 2/9
    always_pos = np.tile([0.0, 0.0, 1.0], (4, 1))
    m = ft.compute_metrics((always_pos, labels))
    assert m["accuracy"] == pytest.approx(0.5)
    assert m["macro_f1"] == pytest.approx(2 / 9)


def test_build_test_report_layout_matches_baseline() -> None:
    # ids: 0=negative 1=neutral 2=positive
    y_true = [2, 2, 1, 1, 0, 0]
    y_pred = [2, 1, 1, 0, 0, 0]
    r = ft.build_test_report(y_true, y_pred)

    assert set(r) == {"accuracy", "macro_f1", "per_class", "confusion_matrix"}
    assert r["accuracy"] == pytest.approx(4 / 6)
    # report order is the baseline's: positive, neutral, negative
    assert r["confusion_matrix"]["labels"] == ["positive", "neutral", "negative"]
    assert r["confusion_matrix"]["matrix"] == [
        [1, 1, 0],  # true positive: 1 right, 1 -> neutral
        [0, 1, 1],  # true neutral: 1 right, 1 -> negative
        [0, 0, 2],  # true negative: both right
    ]
    assert list(r["per_class"]) == ["positive", "neutral", "negative"]
    assert set(r["per_class"]["neutral"]) == {
        "precision",
        "recall",
        "f1-score",
        "support",
    }
    assert r["per_class"]["neutral"]["support"] == 2.0


def test_count_overlap() -> None:
    n = ft.count_overlap(
        train_raw=["Bagus!!", "buruk"],
        test_raw=["bagus!!", "Bagus!!", "baru"],
        train_pre=["bagus!!", "buruk"],
        test_pre=["bagus!!", "bagus!!", "baru"],
    )
    assert n == {
        "test_texts_also_in_train_after_preprocess": 2,
        "test_texts_also_in_train_raw_unique": 1,
    }


# --------------------- epoch bookkeeping / validation ---------------------
def _rows() -> list[dict]:
    return [
        {
            "epoch": 1.0,
            "step": 100,
            "eval_loss": 0.5,
            "eval_accuracy": 0.80,
            "eval_macro_f1": 0.70,
        },
        {
            "epoch": 2.0,
            "step": 200,
            "eval_loss": 0.4,
            "eval_accuracy": 0.85,
            "eval_macro_f1": 0.80,
        },
        {
            "epoch": 3.0,
            "step": 300,
            "eval_loss": 0.45,
            "eval_accuracy": 0.84,
            "eval_macro_f1": 0.78,
        },
    ]


def test_extract_eval_rows_skips_training_logs() -> None:
    history = [{"loss": 1.0, "step": 50}, *_rows(), {"train_runtime": 12.0}]
    assert ft.extract_eval_rows(history) == _rows()


def test_pick_best_index_prefers_trainer_step_then_macro_f1() -> None:
    rows = _rows()
    assert ft.pick_best_index(rows, best_step=200) == 1
    assert ft.pick_best_index(rows, best_step=None) == 1  # argmax macro-F1
    assert ft.pick_best_index(rows, best_step=999) == 1  # unknown step -> fallback
    with pytest.raises(ValueError):
        ft.pick_best_index([], None)


def test_build_validation_section_layout() -> None:
    section = ft.build_validation_section(_rows(), 1, "indobert-base-p1")
    assert set(section) == {"candidates", "selected_index", "reason"}
    assert section["selected_index"] == 1
    assert len(section["candidates"]) == 3
    cand = section["candidates"][1]
    assert set(cand) == {
        "classifier",
        "params",
        "accuracy",
        "macro_f1",
        "log_loss",
        "fit_seconds",
    }
    assert cand["params"] == {"epoch": 2}
    assert cand["macro_f1"] == 0.80
    assert cand["log_loss"] == 0.4
    assert "epoch 2" in section["reason"]


# --------------------------- training arguments ---------------------------
def _cli_args(*extra: str) -> argparse.Namespace:
    return ft.parse_args(list(extra))


def test_default_hyperparameters() -> None:
    a = _cli_args()
    assert a.model_name == "indobenchmark/indobert-base-p1"
    assert (a.max_length, a.learning_rate, a.batch_size, a.epochs) == (128, 2e-5, 32, 4)
    assert (a.weight_decay, a.warmup_ratio, a.seed, a.grad_accum) == (0.01, 0.1, 42, 1)
    assert a.output_dir == ft.DEFAULT_OUTPUT_DIR


def test_training_kwargs_for_new_and_old_transformers(tmp_path) -> None:
    a = _cli_args("--epochs", "2", "--grad-accum", "2")
    new = ft.build_training_kwargs(
        a, tmp_path, fp16=True, supported={"eval_strategy", "warmup_ratio"}
    )
    assert new["eval_strategy"] == "epoch"
    assert new["warmup_ratio"] == 0.1
    assert "evaluation_strategy" not in new
    assert new["num_train_epochs"] == 2
    assert new["gradient_accumulation_steps"] == 2
    assert new["load_best_model_at_end"] is True
    assert new["metric_for_best_model"] == "macro_f1"
    assert new["seed"] == new["data_seed"] == 42
    assert new["fp16"] is True

    old = ft.build_training_kwargs(
        a, tmp_path, fp16=False, supported={"evaluation_strategy"}
    )
    assert old["evaluation_strategy"] == "epoch"
    assert old["warmup_steps"] == 0.1
    assert "warmup_ratio" not in old


# ------------------------------- Dataset -------------------------------
def test_build_dataset_items_and_labels() -> None:
    pytest.importorskip("torch")
    from torch.utils.data import Dataset

    ds = ft.build_dataset(
        ["bagus", "buruk sekali", "biasa"],
        ["positive", "negative", "neutral"],
        _fake_tokenizer,
        max_length=4,
    )
    assert isinstance(ds, Dataset)
    assert len(ds) == 3
    item = ds[1]
    assert item["labels"] == 0  # negative
    assert item["input_ids"] == [ord(c) for c in "buru"]  # truncated to 4
    assert item["attention_mask"] == [1, 1, 1, 1]
    assert ds[0]["labels"] == 2 and ds[2]["labels"] == 1


def test_dataset_rejects_length_mismatch() -> None:
    pytest.importorskip("torch")
    with pytest.raises(ValueError, match="labels"):
        ft.TextClassificationDataset({"input_ids": [[1], [2]]}, [0])
    with pytest.raises(ValueError, match="texts"):
        ft.build_dataset(["a"], ["positive", "negative"], _fake_tokenizer, 4)
