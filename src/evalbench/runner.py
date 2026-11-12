"""Orchestration: generate (resumable) -> score -> cluster -> report -> run.json."""

from __future__ import annotations

import hashlib
import json
import random
import re
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from . import __version__, report
from .failures.cluster import EmbedFn, cluster_failures, minilm_embed
from .models import make_runner
from .schema import Checkpoint, Generation, Item, Verdict, load_suite, suite_hash
from .scorers.judge import Judge, make_judge
from .scorers.score import judge_pairs, score_generations

Log = Callable[[str], None]


def load_config(path: str | Path) -> dict[str, Any]:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    cfg["checkpoints"] = [Checkpoint.model_validate(c) for c in cfg["checkpoints"]]
    return cfg


def _read_jsonl(path: Path, model):
    if not path.exists():
        return []
    return [model.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def generate_all(items: list[Item], ckpts: list[Checkpoint], run_dir: Path, log: Log = print,
                 only: list[str] | None = None, batch_size: int = 12) -> list[Generation]:
    """Generate for every checkpoint. Each checkpoint appends to its own file under generations/,
    so a stopped run resumes where it left off and checkpoints can be generated in parallel processes."""
    gdir = run_dir / "generations"
    gdir.mkdir(parents=True, exist_ok=True)
    out: list[Generation] = []
    for ck in ckpts:
        path = gdir / f"{ck.name}.jsonl"
        done = list({g.item_id: g for g in reversed(_read_jsonl(path, Generation))}.values())[::-1]  # first write wins
        out += done
        if only and ck.name not in only:
            continue
        have = {g.item_id for g in done}
        todo = [i for i in items if i.id not in have]
        if not todo:
            log(f"{ck.name}: all {len(items)} cached")
            continue
        log(f"{ck.name}: generating {len(todo)} of {len(items)}")
        kw = {"batch_size": batch_size} if ck.backend == "hf" else {}
        runner = make_runner(ck, **kw)
        t0 = time.time()
        step = max(batch_size, 6)
        for n in range(0, len(todo), step):
            chunk = runner.generate(todo[n : n + step])
            with open(path, "a", encoding="utf-8") as f:
                for g in chunk:
                    f.write(g.model_dump_json() + "\n")
            out += chunk
            log(f"  {ck.name} {min(n + step, len(todo))}/{len(todo)} ({time.time() - t0:.0f}s)")
        del runner
    return out


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        return ""


def gens_key(items: list[Item], gens: list[Generation]) -> str:
    blob = suite_hash(items) + "\n" + "\n".join(sorted(f"{g.ckpt}|{g.item_id}|{g.text}" for g in gens))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def score_cached(run_dir: Path, items: list[Item], gens: list[Generation], spec: str, log: Log = print):
    """Score with a judge, reusing earlier judge outputs for the same generations (judging is the slow part)."""
    path = run_dir / "judge_cache" / (re.sub(r"\W+", "_", spec) + ".json")
    key = gens_key(items, gens)
    if path.exists():
        c = json.loads(path.read_text(encoding="utf-8"))
        if c["key"] == key:
            log(f"judge {spec}: using cached judgements")
            return [Verdict.model_validate(v) for v in c["verdicts"]], c["pairs"], c["judge"]
    judge = make_judge(spec)
    log(f"judge {spec}: scoring {len(gens)} generations")
    t0 = time.time()
    verdicts, pairs = score_run(items, gens, judge)
    log(f"judge {spec}: done in {time.time() - t0:.0f}s")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"key": key, "judge": judge.name, "verdicts": [v.model_dump() for v in verdicts], "pairs": pairs}), encoding="utf-8")
    return verdicts, pairs, judge.name


def score_run(items: list[Item], gens: list[Generation], judge: Judge | None) -> tuple[list[Verdict], list[dict]]:
    verdicts = score_generations(items, gens, judge)
    pairs = judge_pairs(items, gens, verdicts, judge) if judge is not None else []
    return verdicts, pairs


def build_run_json(items: list[Item], ckpts: list[Checkpoint], gens: list[Generation], verdicts: list[Verdict],
                   pairs: list[dict], judge_name: str, embed: EmbedFn = minilm_embed, seed: int = 2616,
                   diff_pair: list[str] | None = None) -> dict[str, Any]:
    names = [c.name for c in ckpts]
    gi = {(g.ckpt, g.item_id): g for g in gens}
    clusters = cluster_failures(items, gens, verdicts, embed=embed)
    rates = report.pass_rates(verdicts)
    fwr = report.field_win_rate(verdicts)
    modes = report.mode_table(verdicts)
    samples = []
    for v in verdicts:
        g = gi[(v.ckpt, v.item_id)]
        samples.append({"c": names.index(v.ckpt), "i": v.item_id, "p": v.passed, "s": round(v.score, 3), "m": v.mode,
                        "tr": g.truncated, "tx": g.text, "ev": v.evidence, "jp": None if v.judge_p is None else round(v.judge_p, 3),
                        "sc": v.scorer, "sec": g.seconds, "nt": g.new_tokens})
    order = [i.id for i in items]
    random.Random(seed).shuffle(order)
    judge_judge = report.judge_stats(verdicts, pairs)
    base, after = (diff_pair or [names[0], names[-1]])[:2]
    return {
        "meta": {
            "version": __version__, "suite_hash": suite_hash(items), "n_items": len(items), "judge": judge_name,
            "commit": _git_commit(), "created": time.strftime("%Y-%m-%d"),
            "checkpoints": [c.model_dump() for c in ckpts],
            "seconds_per_item": {c: round(sum(g.seconds for g in gens if g.ckpt == c) / max(1, sum(g.ckpt == c for g in gens)), 2) for c in names},
        },
        "items": {i.id: {"task": i.task, "prompt": i.prompt, "gold": i.gold, "source": i.source, "license": i.license,
                         "checker": i.checker} for i in items},
        "samples": samples,
        "order": order,
        "pass_rates": rates.to_dict("records"),
        "field_win": fwr,
        "win_matrix": report.win_matrix(verdicts),
        "mode_table": {c: {m: int(modes.loc[c, m]) if c in modes.index else 0 for m in modes.columns} for c in names},
        "clusters": clusters,
        "judge": judge_judge,
        "targets": report.targets(verdicts),
        "diff": report.diff(verdicts, base, after),
    }


def run_all(config: dict[str, Any], run_dir: str | Path, judge_spec: str | None = None, log: Log = print,
            embed: EmbedFn = minilm_embed, only: list[str] | None = None) -> dict[str, Any]:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    items = load_suite(config["suite"])
    ckpts: list[Checkpoint] = config["checkpoints"]
    gens = generate_all(items, ckpts, run_dir, log, only=only)
    if only:
        return {}
    verdicts, pairs, judge_name = score_cached(run_dir, items, gens, judge_spec or config["judge"], log)
    data = build_run_json(items, ckpts, gens, verdicts, pairs, judge_name, embed=embed, diff_pair=config.get("diff"))
    (run_dir / "verdicts.jsonl").write_text("\n".join(v.model_dump_json() for v in verdicts), encoding="utf-8")
    (run_dir / "run.json").write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    log(f"wrote {run_dir / 'run.json'}")
    return data
