"""Typed records shared by the suite, runner, scorers and API."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

TASKS = ("math", "knowledge", "format", "truth", "unanswerable", "benign")
MODES = ("hallucination", "format_break", "over_refusal", "wrong_answer", "degenerate")


class Item(BaseModel):
    id: str
    task: str
    prompt: str
    checker: str  # exact_numeric | exact_choice | constraints | misconception | abstain | comply
    gold: str | None = None
    rubric: str | None = None  # judge question; {gold} is substituted
    meta: dict[str, Any] = Field(default_factory=dict)
    max_new_tokens: int = 96
    source: str = "authored"
    license: str = "MIT"


class Checkpoint(BaseModel):
    name: str
    model: str
    style: str = "chat"  # chat | plain
    backend: str = "hf"  # hf | stub
    note: str = ""


class Generation(BaseModel):
    ckpt: str
    item_id: str
    text: str
    new_tokens: int
    truncated: bool
    seconds: float


class Verdict(BaseModel):
    ckpt: str
    item_id: str
    task: str
    passed: bool
    score: float
    scorer: str
    evidence: str = ""
    judge_p: float | None = None
    mode: str | None = None  # primary failure mode when not passed
    tags: list[str] = Field(default_factory=list)


def load_suite(path: str | Path) -> list[Item]:
    with open(path, encoding="utf-8") as f:
        return [Item.model_validate_json(line) for line in f if line.strip()]


def suite_hash(items: list[Item]) -> str:
    blob = "\n".join(json.dumps(i.model_dump(), sort_keys=True) for i in items)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]
