from __future__ import annotations

import re


YEAR_BIN_RE = re.compile(r"^\d{4}$")
MONTH_BIN_RE = re.compile(r"^(\d{4})-(\d{2})$")
YEAR_ANYWHERE_RE = re.compile(r"(?<!\d)(\d{4})(?!\d)")
YEAR_MONTH_ANYWHERE_RE = re.compile(r"(?<!\d)(\d{4})[-/](\d{1,2})(?!\d)")


def normalize_granularity(value: str | None, default: str = "year") -> str:
    candidate = str(value or "").strip().lower()
    if not candidate:
        candidate = default
    if candidate not in {"year", "month"}:
        raise ValueError(f"Unsupported time granularity: {value!r}. Use 'year' or 'month'.")
    return candidate


def _valid_year(year: int) -> bool:
    return 1900 <= year <= 2100


def _valid_month(month: int) -> bool:
    return 1 <= month <= 12


def extract_time_bin(value: str, granularity: str = "year") -> str:
    granularity = normalize_granularity(granularity)
    text = str(value).strip()
    if not text:
        return "unknown"

    if granularity == "month":
        month_match = YEAR_MONTH_ANYWHERE_RE.search(text)
        if month_match:
            year = int(month_match.group(1))
            month = int(month_match.group(2))
            if _valid_year(year) and _valid_month(month):
                return f"{year:04d}-{month:02d}"

    year_match = YEAR_ANYWHERE_RE.search(text)
    if year_match:
        year = int(year_match.group(1))
        if _valid_year(year):
            if granularity == "year":
                return f"{year:04d}"
            return f"{year:04d}-01"

    return "unknown"


def classify_time_bin_format(value: str) -> str:
    text = str(value).strip()
    if text == "unknown":
        return "unknown"

    if YEAR_BIN_RE.match(text):
        year = int(text)
        return "year" if _valid_year(year) else "other"

    month_match = MONTH_BIN_RE.match(text)
    if month_match:
        year = int(month_match.group(1))
        month = int(month_match.group(2))
        if _valid_year(year) and _valid_month(month):
            return "month"

    return "other"


def incompatible_time_bins(time_bins: list[str], expected_granularity: str) -> list[str]:
    expected = normalize_granularity(expected_granularity)
    bad = []
    for value in time_bins:
        bucket = classify_time_bin_format(value)
        if bucket in {"unknown", expected}:
            continue
        bad.append(str(value))
    return sorted(set(bad))
