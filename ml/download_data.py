"""Download the SmSA dataset (IndoNLU) into ml/data/smsa/.

Sources are tried in order (no data is ever generated or invented):

1. Hugging Face Hub, dataset ``indonlp/indonlu`` (config ``smsa``), via plain
   HTTP. We look for raw data files (tsv/csv/jsonl) for SmSA in the repo.
   The ``datasets`` library is intentionally NOT used.
2. Official IndoNLU GitHub repository (``IndoNLP/indonlu``),
   ``dataset/smsa_doc-sentiment-prosa/{train,valid,test}_preprocess.tsv``.
   ``test_preprocess.tsv`` is the labelled test file (not the ``masked_label`` one).

If every source fails, the script exits with a non-zero code.

Usage (from the repo root):
    python ml/download_data.py [--force]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ML_DIR = Path(__file__).resolve().parent
DATA_DIR = ML_DIR / "data" / "smsa"
SPLITS = ("train", "valid", "test")
LABELS = {"positive", "neutral", "negative"}

HF_DATASET = "indonlp/indonlu"
HF_TREE_URL = (
    f"https://huggingface.co/api/datasets/{HF_DATASET}/tree/main?recursive=true"
)
HF_RESOLVE_URL = f"https://huggingface.co/datasets/{HF_DATASET}/resolve/main/{{path}}"

GITHUB_BASE = "https://raw.githubusercontent.com/IndoNLP/indonlu/master/dataset/smsa_doc-sentiment-prosa"
GITHUB_FILES = {
    "train": "train_preprocess.tsv",
    "valid": "valid_preprocess.tsv",
    "test": "test_preprocess.tsv",
}

USER_AGENT = "sentiment-dashboard/0.1 (dataset download script)"
TIMEOUT_S = 60
RETRIES = 3


class SourceError(Exception):
    """Raised when a source cannot provide valid data."""


def http_get(url: str) -> bytes:
    last_err: Exception | None = None
    for attempt in range(1, RETRIES + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            # 4xx will not fix itself on retry
            if 400 <= e.code < 500:
                raise SourceError(f"HTTP {e.code} for {url}") from e
            last_err = e
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last_err = e
        if attempt < RETRIES:
            time.sleep(2 * attempt)
    raise SourceError(f"failed to GET {url} after {RETRIES} attempts: {last_err}")


def parse_tsv(raw: bytes) -> list[tuple[str, str]]:
    """Parse SmSA TSV (no header, ``text<TAB>label``) and validate labels."""
    try:
        content = raw.decode("utf-8")
    except UnicodeDecodeError as e:
        raise SourceError(f"file is not valid UTF-8: {e}") from e
    rows: list[tuple[str, str]] = []
    for lineno, line in enumerate(content.splitlines(), start=1):
        if not line.strip():
            continue
        if "\t" not in line:
            raise SourceError(f"line {lineno}: no TAB separator")
        text, label = line.rsplit("\t", 1)
        label = label.strip()
        if label not in LABELS:
            raise SourceError(f"line {lineno}: unexpected label {label!r}")
        rows.append((text, label))
    if not rows:
        raise SourceError("file has no rows")
    return rows


def try_huggingface() -> dict[str, bytes]:
    listing = json.loads(http_get(HF_TREE_URL))
    paths = [item["path"] for item in listing if item.get("type") == "file"]
    data_exts = (".tsv", ".csv", ".jsonl")
    candidates = [
        p for p in paths if "smsa" in p.lower() and p.lower().endswith(data_exts)
    ]
    if not candidates:
        raise SourceError(
            "HF repo contains no raw SmSA data files "
            f"(files present: {', '.join(paths) or 'none'}). "
            "It only ships a loading script, which requires the `datasets` library."
        )
    result: dict[str, bytes] = {}
    for split in SPLITS:
        match = [p for p in candidates if split in Path(p).name.lower()]
        if len(match) != 1 or not match[0].lower().endswith(".tsv"):
            raise SourceError(
                f"could not find a single TSV for split {split!r}: {match}"
            )
        result[split] = http_get(HF_RESOLVE_URL.format(path=match[0]))
    return result


def try_github() -> dict[str, bytes]:
    return {
        split: http_get(f"{GITHUB_BASE}/{name}") for split, name in GITHUB_FILES.items()
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--force", action="store_true", help="re-download even if files exist"
    )
    args = parser.parse_args()

    if not args.force and all((DATA_DIR / f"{s}.tsv").exists() for s in SPLITS):
        print(f"Data already present in {DATA_DIR} (use --force to re-download).")
        return 0

    sources = [
        (
            "huggingface",
            f"https://huggingface.co/datasets/{HF_DATASET} (config smsa)",
            try_huggingface,
        ),
        ("github", GITHUB_BASE, try_github),
    ]
    attempts: list[dict[str, str]] = []
    for name, location, fn in sources:
        print(f"[{name}] trying {location} ...")
        try:
            raw = fn()
            parsed = {split: parse_tsv(content) for split, content in raw.items()}
        except (SourceError, json.JSONDecodeError, KeyError) as e:
            print(f"[{name}] FAILED: {e}")
            attempts.append({"source": name, "status": "failed", "reason": str(e)})
            continue

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        files_info = {}
        for split in SPLITS:
            path = DATA_DIR / f"{split}.tsv"
            path.write_bytes(raw[split])
            counts: dict[str, int] = {}
            for _, label in parsed[split]:
                counts[label] = counts.get(label, 0) + 1
            files_info[split] = {
                "rows": len(parsed[split]),
                "label_counts": dict(sorted(counts.items())),
                "sha256": hashlib.sha256(raw[split]).hexdigest(),
            }
            print(
                f"[{name}] {split}: {len(parsed[split])} rows {files_info[split]['label_counts']}"
            )
        attempts.append({"source": name, "status": "ok", "reason": ""})
        meta = {
            "dataset": "SmSA (IndoNLU)",
            "source": name,
            "location": location,
            "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "attempts": attempts,
            "files": files_info,
        }
        (DATA_DIR / "source.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        print(f"Saved to {DATA_DIR} (source: {name}).")
        return 0

    print("ERROR: all sources failed. No data was written.", file=sys.stderr)
    for a in attempts:
        print(f"  - {a['source']}: {a['reason']}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
