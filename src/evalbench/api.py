"""FastAPI service: serves recorded runs, drill-down queries and a try-a-prompt endpoint."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import __version__
from .models import make_runner
from .runner import load_config
from .schema import Item
from .scorers import constraints, lexicon
from .scorers.extract import extract_number

ROOT = Path(__file__).resolve().parents[2]


class Health(BaseModel):
    status: str
    version: str
    runs: int


class RunInfo(BaseModel):
    id: str
    created: str
    n_items: int
    checkpoints: list[str]
    judge: str


class Sample(BaseModel):
    ckpt: str
    item: str
    task: str
    prompt: str
    gold: str | None
    output: str
    passed: bool
    mode: str | None
    evidence: str
    judge_p: float | None


class TryRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=1200)
    checkpoints: list[str] = Field(min_length=1, max_length=4)
    max_new_tokens: int = Field(default=96, ge=8, le=192)


class TryAnswer(BaseModel):
    ckpt: str
    output: str
    seconds: float
    signals: dict[str, Any]


def create_app(runs_dir: str | Path | None = None, config_path: str | Path | None = None, web_dir: str | Path | None = None) -> FastAPI:
    runs = Path(runs_dir or os.environ.get("EVALBENCH_RUNS", ROOT / "runs"))
    cfg_path = Path(config_path or ROOT / "configs" / "default.json")
    web = Path(web_dir or ROOT / "web")
    app = FastAPI(title="EvalBench", version=__version__)
    runners: dict[str, Any] = {}

    def run_ids() -> list[str]:
        return sorted(p.parent.name for p in runs.glob("*/run.json")) if runs.exists() else []

    @lru_cache(maxsize=4)
    def load(run_id: str, mtime: float) -> dict[str, Any]:
        return json.loads((runs / run_id / "run.json").read_text(encoding="utf-8"))

    def get_run(run_id: str) -> dict[str, Any]:
        p = runs / run_id / "run.json"
        if "/" in run_id or ".." in run_id or not p.exists():
            raise HTTPException(404, "unknown run")
        return load(run_id, p.stat().st_mtime)

    @app.get("/health", response_model=Health)
    def health() -> Health:
        return Health(status="ok", version=__version__, runs=len(run_ids()))

    @app.get("/api/runs", response_model=list[RunInfo])
    def list_runs() -> list[RunInfo]:
        out = []
        for rid in run_ids():
            m = get_run(rid)["meta"]
            out.append(RunInfo(id=rid, created=m["created"], n_items=m["n_items"], judge=m["judge"],
                               checkpoints=[c["name"] for c in m["checkpoints"]]))
        return out

    @app.get("/api/runs/{run_id}")
    def one_run(run_id: str) -> JSONResponse:
        return JSONResponse(get_run(run_id))

    @app.get("/api/runs/{run_id}/samples", response_model=list[Sample])
    def samples(run_id: str, cluster: int | None = None, ckpt: str | None = None, task: str | None = None,
                mode: str | None = None, failed_only: bool = True, limit: int = 200) -> list[Sample]:
        d = get_run(run_id)
        names = [c["name"] for c in d["meta"]["checkpoints"]]
        in_cluster = None
        if cluster is not None:
            in_cluster = {(p["ckpt"], p["item"]) for p in d["clusters"]["points"] if p["cluster"] == cluster}
        out: list[Sample] = []
        for s in d["samples"]:
            name, it = names[s["c"]], d["items"][s["i"]]
            if failed_only and s["p"]:
                continue
            if (ckpt and name != ckpt) or (task and it["task"] != task) or (mode and s["m"] != mode):
                continue
            if in_cluster is not None and (name, s["i"]) not in in_cluster:
                continue
            out.append(Sample(ckpt=name, item=s["i"], task=it["task"], prompt=it["prompt"], gold=it["gold"], output=s["tx"],
                              passed=s["p"], mode=s["m"], evidence=s["ev"], judge_p=s["jp"]))
            if len(out) >= limit:
                break
        return out

    @app.post("/api/try", response_model=list[TryAnswer])
    def try_prompt(req: TryRequest) -> list[TryAnswer]:
        cfg = load_config(cfg_path)
        by_name = {c.name: c for c in cfg["checkpoints"]}
        bad = [c for c in req.checkpoints if c not in by_name]
        if bad:
            raise HTTPException(422, f"unknown checkpoints {bad}")
        item = Item(id="try", task="try", prompt=req.prompt, checker="none", max_new_tokens=req.max_new_tokens)
        answers = []
        for name in req.checkpoints:
            if name not in runners:
                runners[name] = make_runner(by_name[name])
            g = runners[name].generate([item])[0]
            answers.append(TryAnswer(ckpt=name, output=g.text, seconds=g.seconds, signals=signals(g.text)))
        return answers

    @app.get("/data/run.json")
    def latest() -> FileResponse:
        ids = run_ids()
        if not ids:
            raise HTTPException(404, "no runs recorded yet")
        return FileResponse(runs / ids[-1] / "run.json", media_type="application/json")

    @app.get("/data/config.json")
    def config() -> JSONResponse:
        return JSONResponse({"try_enabled": True})

    if web.exists():
        app.mount("/", StaticFiles(directory=web, html=True), name="web")
    return app


def signals(text: str) -> dict[str, Any]:
    val, fenced = constraints.parse_json(text)
    return {
        "refusal_phrase": lexicon.refusal_hit(text),
        "abstention_phrase": lexicon.abstain_hit(text),
        "valid_json": val is not None,
        "words": len(constraints.words(text)),
        "last_number": extract_number(text),
    }


app = create_app()
