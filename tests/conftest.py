from pathlib import Path

import numpy as np
import pytest

from evalbench.runner import load_config, run_all

ROOT = Path(__file__).resolve().parents[1]


def hash_word(w: str) -> int:
    return sum((i + 1) * ord(c) for i, c in enumerate(w))


def fake_embed(texts: list[str]) -> np.ndarray:
    """Deterministic bag-of-words hashing embedding, no model download."""
    out = np.zeros((len(texts), 96))
    for i, t in enumerate(texts):
        for w in t.lower().split():
            out[i, hash_word(w) % 96] += 1
    n = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.maximum(n, 1e-9)


@pytest.fixture(scope="session")
def stub_run(tmp_path_factory):
    d = tmp_path_factory.mktemp("run")
    cfg = load_config(ROOT / "configs" / "stub.json")
    cfg["suite"] = str(ROOT / "data" / "suite.jsonl")
    data = run_all(cfg, d, embed=fake_embed, log=lambda s: None)
    return d, data
