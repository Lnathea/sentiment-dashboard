"""Fine-tune IndoBERT (3 classes) on SmSA. Meant for Google Colab (GPU).

Protocol (no test leakage), mirroring ``ml/train_baseline.py``:
1. Official IndoNLU SmSA splits from ``ml/download_data.py`` (``ml/data/smsa/``),
   used as-is. No new download and no new split.
2. Every text goes through ``ml.preprocess.preprocess`` -- the SAME function the
   backend must use at inference time.
3. Fine-tune on TRAIN. Each epoch is evaluated on VALID; the checkpoint with the
   best macro-F1 on VALID is kept (``load_best_model_at_end``).
4. The best checkpoint is evaluated on TEST exactly ONCE, at the very end.

Label ids of the model (stored in the model config)::

    0 = negative, 1 = neutral, 2 = positive

Reports (``metrics_indobert.json`` and the confusion matrix) use the baseline's
order ``positive, neutral, negative`` so both models can be compared directly.

Usage (from the repo root, after ``python ml/download_data.py``)::

    python ml/finetune_indobert.py [--epochs 4] [--output-dir ml/artifacts/indobert]

Requires torch and transformers (see ``ml/requirements-indobert.txt``). The pure
helper functions in this module work without them.
"""

from __future__ import annotations

import argparse
import inspect
import json
import shutil
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

try:
    import torch
    from torch.utils.data import Dataset as _DatasetBase
except ImportError:  # helpers and their tests still work without torch
    torch = None  # type: ignore[assignment]
    _DatasetBase = object  # type: ignore[assignment,misc]

ML_DIR = Path(__file__).resolve().parent
REPO_ROOT = ML_DIR.parent
if str(REPO_ROOT) not in sys.path:
    # Import like the backend does: ``ml.preprocess``.
    sys.path.insert(0, str(REPO_ROOT))

from ml.preprocess import preprocess_many
from ml.train_baseline import LABELS, load_split, plot_confusion_matrix

BASE_MODEL = "indobenchmark/indobert-base-p1"
DEFAULT_OUTPUT_DIR = ML_DIR / "artifacts" / "indobert"
METRICS_FILENAME = "metrics_indobert.json"
CM_FILENAME = "confusion_matrix_indobert.png"
CHECKPOINT_DIRNAME = "_checkpoints"

# Model label ids. The order is a contract with the backend loader (Fase 2B).
ID2LABEL: dict[int, str] = {0: "negative", 1: "neutral", 2: "positive"}
LABEL2ID: dict[str, int] = {label: i for i, label in ID2LABEL.items()}


# --------------------------------------------------------------------------
# Pure helpers (no torch / transformers needed)
# --------------------------------------------------------------------------
def encode_labels(labels: Sequence[str]) -> list[int]:
    """Map label strings to model ids; raise on anything unexpected."""
    out: list[int] = []
    for label in labels:
        if label not in LABEL2ID:
            raise ValueError(f"unexpected label {label!r}; expected {sorted(LABEL2ID)}")
        out.append(LABEL2ID[label])
    return out


def decode_labels(ids: Sequence[int]) -> list[str]:
    """Map model ids back to label strings."""
    return [ID2LABEL[int(i)] for i in ids]


def compute_metrics(eval_pred: Any) -> dict[str, float]:
    """Trainer ``compute_metrics``: ``eval_pred`` unpacks to (logits, label_ids)."""
    logits, label_ids = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(label_ids, preds)),
        "macro_f1": float(
            f1_score(
                label_ids,
                preds,
                average="macro",
                labels=sorted(ID2LABEL),
                zero_division=0,
            )
        ),
    }


def build_test_report(
    y_true_ids: Sequence[int], y_pred_ids: Sequence[int]
) -> dict[str, Any]:
    """Test section of the metrics file, same layout as the baseline's ``test``."""
    y_true = decode_labels(y_true_ids)
    y_pred = decode_labels(y_pred_ids)
    report = classification_report(
        y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", labels=LABELS, zero_division=0)
        ),
        "per_class": {
            lbl: {k: float(v) for k, v in report[lbl].items()} for lbl in LABELS
        },
        "confusion_matrix": {"labels": list(LABELS), "matrix": cm.tolist()},
    }


def count_overlap(
    train_raw: Sequence[str],
    test_raw: Sequence[str],
    train_pre: Sequence[str],
    test_pre: Sequence[str],
) -> dict[str, int]:
    """Informational only: test texts that also occur in train (same keys as baseline)."""
    train_set = set(train_pre)
    return {
        "test_texts_also_in_train_after_preprocess": sum(
            1 for t in test_pre if t in train_set
        ),
        "test_texts_also_in_train_raw_unique": len(set(test_raw) & set(train_raw)),
    }


def extract_eval_rows(log_history: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Per-epoch VALID evaluations from ``trainer.state.log_history``."""
    return [dict(e) for e in log_history if "eval_macro_f1" in e]


def pick_best_index(rows: Sequence[Mapping[str, Any]], best_step: int | None) -> int:
    """Index of the epoch whose checkpoint the Trainer kept.

    Uses the Trainer's own ``best_global_step`` when it matches an evaluation;
    otherwise falls back to the highest macro-F1 (ties: accuracy, then log-loss).
    """
    if not rows:
        raise ValueError("no validation evaluations were recorded")
    if best_step is not None:
        for i, row in enumerate(rows):
            if row.get("step") == best_step:
                return i
    return max(
        range(len(rows)),
        key=lambda i: (
            rows[i]["eval_macro_f1"],
            rows[i]["eval_accuracy"],
            -rows[i]["eval_loss"],
        ),
    )


def build_validation_section(
    rows: Sequence[Mapping[str, Any]], best_index: int, model_name: str
) -> dict[str, Any]:
    """Validation section, same layout as the baseline's ``validation``.

    One candidate per epoch. ``log_loss`` is the mean cross-entropy on VALID
    (Trainer's ``eval_loss``). ``fit_seconds`` is not measured per epoch -> null.
    """
    candidates = [
        {
            "classifier": model_name,
            "params": {"epoch": round(row["epoch"])},
            "accuracy": float(row["eval_accuracy"]),
            "macro_f1": float(row["eval_macro_f1"]),
            "log_loss": float(row["eval_loss"]),
            "fit_seconds": None,
        }
        for row in rows
    ]
    best = candidates[best_index]
    reason = (
        f"Selected epoch {best['params']['epoch']} of {len(candidates)} because it "
        f"has the highest macro-F1 on the validation split ({best['macro_f1']:.4f}), "
        "via load_best_model_at_end. Macro-F1 is used because the classes are "
        "imbalanced. TEST was evaluated once, after selection."
    )
    return {
        "candidates": candidates,
        "selected_index": best_index,
        "reason": reason,
    }


class TextClassificationDataset(_DatasetBase):  # type: ignore[misc,valid-type]
    """Tokenized texts + integer labels; padding is left to the data collator."""

    def __init__(self, encodings: Mapping[str, Sequence[Any]], labels: Sequence[int]):
        self.encodings = {k: list(v) for k, v in encodings.items()}
        self.labels = [int(label) for label in labels]
        for key, values in self.encodings.items():
            if len(values) != len(self.labels):
                raise ValueError(
                    f"encodings[{key!r}] has {len(values)} items "
                    f"but there are {len(self.labels)} labels"
                )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def build_dataset(
    texts: Sequence[str],
    labels: Sequence[str],
    tokenizer: Callable[..., Mapping[str, Sequence[Any]]],
    max_length: int,
) -> TextClassificationDataset:
    """Tokenize (truncation only) and wrap with integer labels."""
    if len(texts) != len(labels):
        raise ValueError(f"{len(texts)} texts but {len(labels)} labels")
    encodings = tokenizer(list(texts), truncation=True, max_length=max_length)
    return TextClassificationDataset(dict(encodings), encode_labels(labels))


def build_training_kwargs(
    args: argparse.Namespace,
    checkpoint_dir: Path,
    fp16: bool,
    supported: set[str],
) -> dict[str, Any]:
    """TrainingArguments kwargs, adapted to the installed transformers version.

    Two names changed across transformers versions: ``evaluation_strategy`` ->
    ``eval_strategy`` and ``warmup_ratio`` (newer versions accept a float
    ``warmup_steps`` < 1 as a ratio). ``supported`` is the set of accepted names.
    """
    kwargs: dict[str, Any] = {
        "output_dir": str(checkpoint_dir),
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
        "logging_steps": 50,
        "seed": args.seed,
        "data_seed": args.seed,
        "fp16": fp16,
        "report_to": "none",
        "dataloader_num_workers": 0,
    }
    kwargs[
        "eval_strategy" if "eval_strategy" in supported else "evaluation_strategy"
    ] = "epoch"
    if "warmup_ratio" in supported:
        kwargs["warmup_ratio"] = args.warmup_ratio
    else:
        kwargs["warmup_steps"] = args.warmup_ratio
    return kwargs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune IndoBERT on SmSA (3 classes).")
    p.add_argument("--model-name", default=BASE_MODEL, help="base model (HF Hub id)")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument(
        "--batch-size", type=int, default=32, help="per-device train batch size"
    )
    p.add_argument(
        "--grad-accum",
        type=int,
        default=1,
        help="gradient accumulation steps (effective batch = batch-size * grad-accum)",
    )
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-fp16", action="store_true", help="disable fp16 even on GPU")
    p.add_argument(
        "--keep-checkpoints",
        action="store_true",
        help=f"keep <output-dir>/{CHECKPOINT_DIRNAME} (large; deleted by default)",
    )
    return p.parse_args(argv)


# --------------------------------------------------------------------------
# Training (needs torch + transformers)
# --------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if torch is None:
        raise SystemExit(
            "torch is not installed. On Colab it is preinstalled; elsewhere install it "
            "first, then `pip install -r ml/requirements-indobert.txt`."
        )
    import transformers
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    t0 = time.time()
    set_seed(args.seed)  # python, numpy, torch (+ cuda)
    cuda = torch.cuda.is_available()
    device = torch.cuda.get_device_name(0) if cuda else "cpu"
    fp16 = cuda and not args.no_fp16
    print(f"Device: {device} | fp16={fp16}")
    if not cuda:
        print("WARNING: no GPU found. Training on CPU will be extremely slow.")

    # ---- data: same official splits and same preprocess() as the baseline ----
    train_raw, y_train = load_split("train")
    valid_raw, y_valid = load_split("valid")
    test_raw, y_test = load_split("test")
    print(f"Rows: train={len(y_train)} valid={len(y_valid)} test={len(y_test)}")
    train_pre = preprocess_many(train_raw)
    valid_pre = preprocess_many(valid_raw)
    test_pre = preprocess_many(test_raw)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_ds = build_dataset(train_pre, y_train, tokenizer, args.max_length)
    valid_ds = build_dataset(valid_pre, y_valid, tokenizer, args.max_length)
    test_ds = build_dataset(test_pre, y_test, tokenizer, args.max_length)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(ID2LABEL),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = out_dir / CHECKPOINT_DIRNAME

    supported = set(inspect.signature(TrainingArguments.__init__).parameters)
    training_args = TrainingArguments(
        **build_training_kwargs(args, checkpoint_dir, fp16, supported)
    )
    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": train_ds,
        "eval_dataset": valid_ds,
        "data_collator": DataCollatorWithPadding(tokenizer=tokenizer),
        "compute_metrics": compute_metrics,
    }
    if "processing_class" in inspect.signature(Trainer.__init__).parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:  # older transformers
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = Trainer(**trainer_kwargs)

    # ---- fine-tune; best epoch chosen on VALID macro-F1 ----
    trainer.train()
    rows = extract_eval_rows(trainer.state.log_history)
    best_step = getattr(trainer.state, "best_global_step", None)
    best_idx = pick_best_index(rows, best_step)
    validation = build_validation_section(rows, best_idx, "indobert-base-p1")
    best_epoch = validation["candidates"][best_idx]["params"]["epoch"]
    best_valid = validation["candidates"][best_idx]
    print(
        f"Best epoch (valid): {best_epoch} | macroF1={best_valid['macro_f1']:.4f} "
        f"acc={best_valid['accuracy']:.4f}"
    )

    # ---- final evaluation on TEST (used exactly once) ----
    pred = trainer.predict(test_ds)
    y_test_ids = encode_labels(y_test)
    y_pred_ids = np.argmax(pred.predictions, axis=-1).tolist()
    test_report = build_test_report(y_test_ids, y_pred_ids)

    # ---- save: model + tokenizer (safetensors), metrics, confusion matrix ----
    best_model = trainer.model
    if best_model.config.id2label != ID2LABEL:
        raise RuntimeError(
            f"id2label mismatch in model config: {best_model.config.id2label}"
        )
    best_model.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)
    if not args.keep_checkpoints:
        shutil.rmtree(checkpoint_dir, ignore_errors=True)

    metrics = {
        "dataset": "SmSA (IndoNLU), official splits",
        "split_sizes": {
            "train": len(y_train),
            "valid": len(y_valid),
            "test": len(y_test),
        },
        "label_order": list(LABELS),
        "hyperparameters": {
            "base_model": args.model_name,
            "max_length": args.max_length,
            "learning_rate": args.learning_rate,
            "per_device_train_batch_size": args.batch_size,
            "gradient_accumulation_steps": args.grad_accum,
            "effective_batch_size": args.batch_size * args.grad_accum,
            "epochs": args.epochs,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "fp16": fp16,
            "seed": args.seed,
            "metric_for_best_model": "macro_f1 (validation)",
        },
        "validation": validation,
        "test": test_report,
        "data_notes": count_overlap(train_raw, test_raw, train_pre, test_pre),
        "model": {
            "model_name": "indobert",
            "base_model": args.model_name,
            "id2label": {str(k): v for k, v in ID2LABEL.items()},
            "labels": [ID2LABEL[i] for i in sorted(ID2LABEL)],
            "best_epoch": best_epoch,
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "sklearn_version": sklearn.__version__,
            "device": device,
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "preprocess": "ml.preprocess.preprocess (applied before tokenization)",
            "random_state": args.seed,
        },
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    (out_dir / METRICS_FILENAME).write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    title = (
        f"SmSA test - indobert (best epoch {best_epoch})\n"
        f"acc={test_report['accuracy']:.3f}  macro-F1={test_report['macro_f1']:.3f}"
    )
    plot_confusion_matrix(
        np.array(test_report["confusion_matrix"]["matrix"]),
        out_dir / CM_FILENAME,
        title,
    )

    # ---- summary ----
    print("\n=== Summary ===")
    print(f"Best valid macro-F1 (epoch {best_epoch}): {best_valid['macro_f1']:.4f}")
    print(
        f"TEST  | acc={test_report['accuracy']:.4f} "
        f"macroF1={test_report['macro_f1']:.4f}"
    )
    for lbl in LABELS:
        print(f"  F1 {lbl:<9}: {test_report['per_class'][lbl]['f1-score']:.4f}")
    overlap = metrics["data_notes"]["test_texts_also_in_train_after_preprocess"]
    print(
        f"Test texts also present in train (after preprocess): {overlap}/{len(y_test)}"
    )
    print(f"Saved to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
