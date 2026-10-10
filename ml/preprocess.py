"""Text preprocessing shared by training (ml/) and inference (backend/).

IMPORTANT: this module is imported by the backend (``from ml.preprocess import
preprocess``). Do not copy it elsewhere; any change here affects both training
and inference, so retrain the model after changing it.

Only the Python standard library is used on purpose.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

__all__ = ["preprocess", "preprocess_many"]

_URL_RE = re.compile(r"(?:https?://|www\.)\S+", flags=re.IGNORECASE)
# A mention is "@" + word characters, not preceded by a word character
# (so e-mail addresses like a@b.com are not treated as mentions).
_MENTION_RE = re.compile(r"(?<!\w)@\w+")
_HASHTAG_SYMBOL_RE = re.compile(r"#(?=\w)")
# Any character repeated 3+ times is shortened to 2 ("bangeeet" -> "bangeet",
# "!!!!" -> "!!"). Two repeats are kept because Indonesian words such as
# "maaf" or "saat" legitimately contain double letters.
_REPEAT_RE = re.compile(r"(.)\1{2,}", flags=re.DOTALL)
_WHITESPACE_RE = re.compile(r"\s+")


def preprocess(text: str) -> str:
    """Normalize one text.

    Steps (in order):
    1. Unicode NFKC normalization (e.g. full-width chars, ligatures).
    2. Lowercase.
    3. Remove URLs (``http://``, ``https://``, ``www.``).
    4. Remove ``@mentions``.
    5. Remove the ``#`` symbol of hashtags but keep the word (``#keren`` -> ``keren``).
    6. Shorten characters repeated 3+ times to 2.
    7. Collapse whitespace and strip.
    """
    if not isinstance(text, str):
        raise TypeError(f"preprocess() expects str, got {type(text).__name__}")
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = _URL_RE.sub(" ", text)
    text = _MENTION_RE.sub(" ", text)
    text = _HASHTAG_SYMBOL_RE.sub("", text)
    text = _REPEAT_RE.sub(r"\1\1", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def preprocess_many(texts: Iterable[str]) -> list[str]:
    """Apply :func:`preprocess` to every text."""
    return [preprocess(t) for t in texts]
