"""Pandas reports: pass rates, pairwise win-rates, failure-mode tables, run diffs, judge reliability."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from .schema import MODES, Verdict


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def frame(verdicts: list[Verdict]) -> pd.DataFrame:
    return pd.DataFrame([v.model_dump() for v in verdicts])


def pass_rates(verdicts: list[Verdict]) -> pd.DataFrame:
    df = frame(verdicts)
    rows = []
    for (ck, task), g in list(df.groupby(["ckpt", "task"])) + [((c, "ALL"), g) for c, g in df.groupby("ckpt")]:
        n, k = len(g), int(g.passed.sum())
        lo, hi = wilson(k, n)
        rows.append({"ckpt": ck, "task": task, "n": n, "passed": k, "rate": k / n, "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows)


def _score_matrix(verdicts: list[Verdict]) -> pd.DataFrame:
    df = frame(verdicts)
    return df.pivot(index="item_id", columns="ckpt", values="score")


def win_matrix(verdicts: list[Verdict]) -> dict[str, Any]:
    """Per item, the higher score wins; equal scores are ties and count half a win."""
    m = _score_matrix(verdicts)
    cks = list(m.columns)
    cell: dict[str, dict[str, dict[str, float]]] = {}
    for a in cks:
        cell[a] = {}
        for b in cks:
            if a == b:
                continue
            w = int((m[a] > m[b]).sum())
            lo = int((m[a] < m[b]).sum())
            t = int((m[a] == m[b]).sum())
            cell[a][b] = {"wins": w, "losses": lo, "ties": t, "win_rate": (w + 0.5 * t) / len(m),
                          "decisive": w / (w + lo) if w + lo else 0.5}
    return {"ckpts": cks, "cells": cell}


def field_win_rate(verdicts: list[Verdict], boot: int = 1000, seed: int = 11) -> dict[str, dict[str, float]]:
    """Win-rate of each checkpoint against all the others, ties as half, with a bootstrap 95% CI over items."""
    m = _score_matrix(verdicts)
    rng = np.random.default_rng(seed)
    arr = m.to_numpy()
    n, k = arr.shape

    def rates(a: np.ndarray) -> np.ndarray:
        out = np.zeros(k)
        for j in range(k):
            others = np.delete(a, j, axis=1)
            out[j] = ((a[:, [j]] > others).sum() + 0.5 * (a[:, [j]] == others).sum()) / (others.size)
        return out

    point = rates(arr)
    bs = np.array([rates(arr[rng.integers(0, n, n)]) for _ in range(boot)])
    return {c: {"win_rate": float(point[j]), "ci_lo": float(np.percentile(bs[:, j], 2.5)),
                "ci_hi": float(np.percentile(bs[:, j], 97.5))} for j, c in enumerate(m.columns)}


def mode_table(verdicts: list[Verdict]) -> pd.DataFrame:
    df = frame(verdicts)
    f = df[~df.passed]
    t = f.groupby(["ckpt", "mode"]).size().unstack(fill_value=0)
    return t.reindex(columns=list(MODES), fill_value=0)


def diff(verdicts: list[Verdict], a: str, b: str) -> dict[str, Any]:
    """What changed going from checkpoint a to checkpoint b on the same items."""
    va = {v.item_id: v for v in verdicts if v.ckpt == a}
    vb = {v.item_id: v for v in verdicts if v.ckpt == b}
    ids = sorted(set(va) & set(vb))
    fixed = [i for i in ids if not va[i].passed and vb[i].passed]
    regressed = [i for i in ids if va[i].passed and not vb[i].passed]
    both_fail = [i for i in ids if not va[i].passed and not vb[i].passed]
    ma = pd.Series([v.mode for v in va.values() if not v.passed]).value_counts().reindex(list(MODES), fill_value=0)
    mb = pd.Series([v.mode for v in vb.values() if not v.passed]).value_counts().reindex(list(MODES), fill_value=0)
    return {
        "from": a, "to": b, "n": len(ids), "fixed": fixed, "regressed": regressed, "still_failing": both_fail,
        "modes_from": {k: int(v) for k, v in ma.items()}, "modes_to": {k: int(v) for k, v in mb.items()},
        "modes_delta": {k: int(mb[k] - ma[k]) for k in MODES},
    }


def kappa(y: list[bool], p: list[bool]) -> float:
    n = len(y)
    if n == 0:
        return 0.0
    po = sum(a == b for a, b in zip(y, p, strict=True)) / n
    py, pp = sum(y) / n, sum(p) / n
    pe = py * pp + (1 - py) * (1 - pp)
    return 0.0 if pe == 1 else (po - pe) / (1 - pe)


def judge_stats(verdicts: list[Verdict], pairs: list[dict] | None = None, threshold: float = 0.5) -> dict[str, Any]:
    """Judge reliability against deterministic labels.

    math and knowledge: gold = exact-match result (a strict label). unanswerable and benign: the
    lexicon label (weaker gold, an agreement figure). pairs: pairwise accuracy and position-flip rate."""
    out: dict[str, Any] = {}
    strict = [v for v in verdicts if v.task in ("math", "knowledge") and v.judge_p is not None]
    if strict:
        y = [v.passed for v in strict]
        p = [v.judge_p >= threshold for v in strict]
        tp = sum(a and b for a, b in zip(y, p, strict=True))
        out["strict"] = {
            "n": len(y), "accuracy": sum(a == b for a, b in zip(y, p, strict=True)) / len(y), "kappa": kappa(y, p),
            "gold_pass_rate": sum(y) / len(y), "judge_yes_rate": sum(p) / len(p),
            "precision": tp / max(sum(p), 1), "recall": tp / max(sum(y), 1),
        }
    weak = [v for v in verdicts if v.task in ("unanswerable", "benign") and v.judge_p is not None]
    if weak:
        # judge question: unanswerable asks "admits not knowing" (yes == pass); benign asks "refuses" (yes == fail)
        y = [v.passed for v in weak]
        p = [(v.judge_p >= threshold) == (v.task == "unanswerable") for v in weak]
        out["lexicon_agreement"] = {"n": len(y), "accuracy": sum(a == b for a, b in zip(y, p, strict=True)) / len(y),
                                    "kappa": kappa(y, p)}
    if pairs:
        acc_ab = [(r["p_a_first"] >= 0.5) == (r["gold"] == r["a"]) for r in pairs]
        acc_ba = [(r["p_a_second"] >= 0.5) == (r["gold"] == r["a"]) for r in pairs]
        flips = [(r["p_a_first"] >= 0.5) != (r["p_a_second"] >= 0.5) for r in pairs]
        consistent = [
            ((r["p_a_first"] + r["p_a_second"]) / 2 >= 0.5) == (r["gold"] == r["a"]) for r in pairs
        ]
        out["pairwise"] = {"n": len(pairs), "accuracy_order1": float(np.mean(acc_ab)), "accuracy_order2": float(np.mean(acc_ba)),
                           "flip_rate": float(np.mean(flips)), "accuracy_swap_averaged": float(np.mean(consistent))}
    return out


def targets(verdicts: list[Verdict], per: int = 3) -> dict[str, list[dict[str, Any]]]:
    """Ranked failure modes per checkpoint with example items: input for the next fine-tuning cycle."""
    df = frame(verdicts)
    f = df[~df.passed]
    out: dict[str, list[dict[str, Any]]] = {}
    for ck, g in f.groupby("ckpt"):
        rows = []
        for mode, gg in g.groupby("mode"):
            rows.append({"mode": mode, "count": int(len(gg)), "tasks": sorted(gg.task.unique().tolist()),
                         "examples": gg.item_id.head(per).tolist()})
        out[ck] = sorted(rows, key=lambda r: -r["count"])
    return out


def cap_hits(gens: list[Any], items: dict[str, str]) -> dict[str, Any]:
    """Share of generations that ran to the token cap, per checkpoint and per task (items maps item id to task).

    A hit whose text is a loop (same detector as the "degenerate" mode) is counted as a loop: more tokens would not
    have finished it. The remaining hits are plain budget exhaustion, the part that is an artefact of the cap."""
    from .scorers.score import _repetitive

    out: dict[str, Any] = {}
    for ck in sorted({g.ckpt for g in gens}):
        mine = [g for g in gens if g.ckpt == ck]
        hits = [g for g in mine if g.truncated]
        loops = sum(_repetitive(g.text) for g in hits)
        by_task: dict[str, list[int]] = {}
        for g in mine:
            by_task.setdefault(items[g.item_id], []).append(int(g.truncated))
        out[ck] = {"n": len(mine), "hits": len(hits), "rate": len(hits) / len(mine), "loops": loops,
                   "budget_only": len(hits) - loops, "budget_rate": (len(hits) - loops) / len(mine),
                   "by_task": {t: {"n": len(v), "hits": sum(v)} for t, v in sorted(by_task.items())}}
    return out
