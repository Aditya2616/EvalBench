"""Deterministic verifiers for format-following constraints (IFEval-style, reimplemented)."""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.S)


def words(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text)


def parse_json(text: str) -> tuple[Any | None, bool]:
    """Return (value, fenced). Accepts a single markdown fence but nothing else around the JSON."""
    s = text.strip()
    fenced = False
    m = _FENCE.match(s)
    if m:
        s, fenced = m.group(1), True
    try:
        return json.loads(s), fenced
    except (ValueError, TypeError):
        return None, fenced


def check(c: dict[str, Any], text: str) -> tuple[bool, str]:
    t, kind = text.strip(), c["type"]
    if kind == "json_keys":
        val, _ = parse_json(t)
        if not isinstance(val, dict):
            return False, "output is not a single valid JSON object"
        missing = [k for k in c["keys"] if k not in val]
        return (not missing, f"missing keys {missing}" if missing else "valid JSON with required keys")
    if kind == "no_comma":
        return ("," not in t, "contains a comma" if "," in t else "no commas")
    if kind == "lowercase":
        ok = t == t.lower() and any(ch.isalpha() for ch in t)
        return ok, "all lowercase" if ok else "contains capital letters"
    if kind == "uppercase":
        ok = t == t.upper() and any(ch.isalpha() for ch in t)
        return ok, "all capitals" if ok else "contains lowercase letters"
    if kind == "n_bullets":
        n = len(re.findall(r"^\s*[-*] ", t, re.M))
        return n == c["n"], f"{n} bullets, wanted {c['n']}"
    if kind == "n_highlights":
        n = len(re.findall(r"\*[^\n*]+\*", t))
        return n >= c["n"], f"{n} highlighted sections, wanted at least {c['n']}"
    if kind == "min_words":
        n = len(words(t))
        return n >= c["n"], f"{n} words, wanted at least {c['n']}"
    if kind == "max_words":
        n = len(words(t))
        return n <= c["n"], f"{n} words, wanted at most {c['n']}"
    if kind == "end_phrase":
        ok = t.endswith(c["phrase"])
        return ok, "ends with required phrase" if ok else "does not end with required phrase"
    if kind == "wrap_quotes":
        ok = len(t) > 1 and t[0] == '"' and t[-1] == '"'
        return ok, "wrapped in quotes" if ok else "not wrapped in double quotes"
    if kind == "title":
        ok = re.search(r"<<[^\n]+>>", t) is not None
        return ok, "has <<title>>" if ok else "no <<title>>"
    if kind == "forbidden":
        hit = [w for w in c["words"] if re.search(rf"\b{re.escape(w)}\b", t, re.I)]
        return not hit, f"uses forbidden words {hit}" if hit else "no forbidden words"
    if kind == "keywords":
        miss = [w for w in c["keywords"] if not re.search(re.escape(w), t, re.I)]
        return not miss, f"missing keywords {miss}" if miss else "has keywords"
    raise ValueError(f"unknown constraint {kind}")


def check_all(cs: list[dict[str, Any]], text: str) -> tuple[float, bool, str]:
    results = [check(c, text) for c in cs]
    frac = sum(ok for ok, _ in results) / len(results)
    return frac, all(ok for ok, _ in results), "; ".join(msg for _, msg in results)
