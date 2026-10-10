"""CSV upload parsing and validation (stdlib csv, no pandas)."""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field
from datetime import date

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class CSVFormatError(ValueError):
    """The file cannot be read as a UTF-8 CSV (-> HTTP 400)."""


class CSVContentError(ValueError):
    """The CSV is readable but its content is invalid (-> HTTP 422)."""


@dataclass
class ParsedRow:
    text: str
    source_date: date | None


@dataclass
class ParsedCSV:
    rows: list[ParsedRow] = field(default_factory=list)
    skipped_too_long: int = 0
    ignored_empty: int = 0


def parse_csv(content: bytes, max_text_length: int) -> ParsedCSV:
    """Parse an uploaded CSV.

    - Required column ``text``; optional column ``date`` (YYYY-MM-DD).
      Header names are matched case-insensitively after trimming spaces.
    - Rows whose text is empty/whitespace are ignored (not counted as skipped).
    - Rows whose text is longer than ``max_text_length`` characters are skipped
      and counted.
    - Any non-empty ``date`` that is not a valid YYYY-MM-DD date rejects the
      whole file (the error message contains the line number).
    """
    if not content.strip():
        raise CSVFormatError("Uploaded file is empty.")
    try:
        decoded = content.decode("utf-8-sig")  # tolerate a UTF-8 BOM (Excel)
    except UnicodeDecodeError as e:
        raise CSVFormatError(
            f"File is not valid UTF-8 (invalid byte at position {e.start}). "
            "Save the CSV with UTF-8 encoding."
        ) from e
    if "\x00" in decoded:
        raise CSVFormatError(
            "File contains NUL bytes; it does not look like a text CSV."
        )

    reader = csv.reader(io.StringIO(decoded, newline=""))
    try:
        header = next(reader)
    except StopIteration as e:  # pragma: no cover - guarded by the empty check above
        raise CSVFormatError("Uploaded file is empty.") from e
    except csv.Error as e:
        raise CSVFormatError(f"Could not parse CSV header: {e}") from e

    columns = [h.strip().lower() for h in header]
    if "text" not in columns:
        found = ", ".join(repr(c) for c in columns if c) or "none"
        raise CSVContentError(
            f"CSV must contain a 'text' column. Columns found: {found}."
        )
    text_idx = columns.index("text")
    date_idx = columns.index("date") if "date" in columns else None

    result = ParsedCSV()
    try:
        for row in reader:
            line = reader.line_num
            text = row[text_idx].strip() if text_idx < len(row) else ""
            if not text:
                result.ignored_empty += 1
                continue
            source_date = None
            if date_idx is not None and date_idx < len(row) and row[date_idx].strip():
                source_date = _parse_date(row[date_idx].strip(), line)
            if len(text) > max_text_length:
                result.skipped_too_long += 1
                continue
            result.rows.append(ParsedRow(text=text, source_date=source_date))
    except csv.Error as e:
        raise CSVFormatError(
            f"Could not parse CSV near line {reader.line_num}: {e}"
        ) from e
    return result


def _parse_date(value: str, line: int) -> date:
    if _DATE_RE.match(value):
        try:
            return date.fromisoformat(value)
        except ValueError:
            pass
    raise CSVContentError(
        f"Invalid date {value!r} on line {line}: expected format YYYY-MM-DD."
    )
