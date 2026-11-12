"""Answer extraction for exact-match scoring (regex filters, as in lm-evaluation-harness)."""

from __future__ import annotations

import re

_NUM = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?")


def _to_float(s: str) -> float | None:
    try:
        return float(s.replace(",", "").replace("$", ""))
    except ValueError:
        return None


def extract_number(text: str) -> float | None:
    """Prefer an explicit 'Answer: x'; otherwise the last number in the text."""
    m = re.findall(r"answer\s*(?:is|:)?\s*[:=]?\s*(-?\$?\d[\d,]*(?:\.\d+)?)", text, re.I)
    if m:
        return _to_float(m[-1])
    nums = _NUM.findall(text)
    return _to_float(nums[-1]) if nums else None


def extract_choice(text: str) -> str | None:
    t = text.strip()
    m = re.match(r"^\(?([ABCD])\b[).:]?", t)
    if m:
        return m.group(1)
    m = re.search(r"answer\s*(?:is|:)?\s*\(?([ABCD])\b", t, re.I)
    if m:
        return m.group(1).upper()
    m = re.findall(r"\b([ABCD])\b[).]", t)
    return m[0] if m else None
