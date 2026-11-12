"""Command line interface: evalbench suite|run|calibrate|report|diff|targets|serve|export-demo."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from . import report
from .runner import generate_all, load_config, run_all, score_cached
from .schema import Verdict, load_suite


def _load_run(path: str):
    d = json.loads((Path(path) / "run.json").read_text(encoding="utf-8"))
    return d


def cmd_suite(a: argparse.Namespace) -> None:
    from .suite.build import build_suite

    build_suite(a.out)


def cmd_run(a: argparse.Namespace) -> None:
    cfg = load_config(a.config)
    run_all(cfg, a.out, judge_spec=a.judge, only=a.only)


def cmd_calibrate(a: argparse.Namespace) -> None:
    """Score the recorded generations with each candidate judge and compare it with the deterministic labels."""
    cfg = load_config(a.config)
    items = load_suite(cfg["suite"])
    run_dir = Path(a.run)
    gens = generate_all(items, cfg["checkpoints"], run_dir, print)
    res = {}
    for spec in a.judges:
        verdicts, pairs, _ = score_cached(run_dir, items, gens, spec)
        res[spec] = report.judge_stats(verdicts, pairs)
        print(spec, json.dumps(res[spec], indent=1))
    best = max(res, key=lambda s: res[s].get("strict", {}).get("kappa", -1))
    out = {"judges": res, "best_by_strict_kappa": best}
    (run_dir / "judge_calibration.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("best judge by kappa on exact-match labels:", best)


def cmd_report(a: argparse.Namespace) -> None:
    d = _load_run(a.run)
    names = [c["name"] for c in d["meta"]["checkpoints"]]
    print(f"run {a.run}: {d['meta']['n_items']} items, judge {d['meta']['judge']}, suite {d['meta']['suite_hash']}\n")
    print("pass rate by checkpoint (95% Wilson CI)")
    for r in d["pass_rates"]:
        if r["task"] == "ALL":
            print(f"  {r['ckpt']:<20} {r['passed']:>3}/{r['n']:<3} {r['rate']*100:5.1f}%  [{r['ci_lo']*100:.1f}, {r['ci_hi']*100:.1f}]")
    print("\nwin-rate against the field (bootstrap 95% CI)")
    for c in names:
        w = d["field_win"][c]
        print(f"  {c:<20} {w['win_rate']:.3f}  [{w['ci_lo']:.3f}, {w['ci_hi']:.3f}]")
    print("\nfailure modes (counts)")
    for c in names:
        print(f"  {c:<20} " + "  ".join(f"{m}={n}" for m, n in d["mode_table"][c].items()))
    print("\ncap-hit rate (generation ran to its token limit)")
    for c in names:
        h = d["meta"]["cap_hits"][c]
        print(f"  {c:<20} {h['hits']:>3}/{h['n']:<3} {h['rate']*100:4.1f}%  of which loops {h['loops']}, budget only {h['budget_only']} ({h['budget_rate']*100:.1f}%)")
    st = d["clusters"].get("stats", {})
    print("\nclustering of failures (k by silhouette; purity and AMI against the detector modes)")
    for name, r in st.get("representations", {}).items():
        mark = "  <- used" if name == st.get("chosen") else ""
        print(f"  {name:<13} k={r['k']:<2} silhouette={r['silhouette']:.3f} purity={r['purity']:.3f} "
              f"AMI(mode)={r['ami_vs_mode']:.3f} AMI(task)={r['ami_vs_task']:.3f} disagree={r['mislabelled']}{mark}")
    for isl in d["clusters"].get("islands", []):
        subs = [c for c in d["clusters"]["clusters"] if c["id"] in isl["subs"]]
        print(f"  {isl['mode']:<14} n={isl['n']:<3} disagree={isl['disagree']:<3} " + " | ".join(f"{c['name']} ({c['size']})" for c in subs))


def cmd_diff(a: argparse.Namespace) -> None:
    d = _load_run(a.run)
    verdicts = [
        Verdict(ckpt=d["meta"]["checkpoints"][s["c"]]["name"], item_id=s["i"], task=d["items"][s["i"]]["task"], passed=s["p"],
                score=s["s"], scorer=s["sc"], mode=s["m"]) for s in d["samples"]
    ]
    x = report.diff(verdicts, a.a, a.b)
    print(f"{a.a} -> {a.b} on {x['n']} items: fixed {len(x['fixed'])}, regressed {len(x['regressed'])}, still failing {len(x['still_failing'])}")
    for m in x["modes_delta"]:
        print(f"  {m:<14} {x['modes_from'][m]:>3} -> {x['modes_to'][m]:>3}  ({x['modes_delta'][m]:+d})")


def cmd_targets(a: argparse.Namespace) -> None:
    d = _load_run(a.run)
    with open(a.out, "w", encoding="utf-8") as f:
        for ck, rows in d["targets"].items():
            for r in rows:
                f.write(json.dumps({"checkpoint": ck, **r}) + "\n")
    print(f"wrote {a.out}")


def cmd_serve(a: argparse.Namespace) -> None:
    import uvicorn

    uvicorn.run("evalbench.api:app", host=a.host, port=a.port)


def cmd_export(a: argparse.Namespace) -> None:
    root = Path(a.docs)
    web = Path(a.web)
    if root.exists():
        shutil.rmtree(root / "data", ignore_errors=True)
    for p in web.rglob("*"):
        if p.is_file():
            dest = root / p.relative_to(web)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest)
    (root / "data").mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(a.run) / "run.json", root / "data" / "run.json")
    (root / "data" / "config.json").write_text(json.dumps({"try_enabled": False}), encoding="utf-8")
    (root / ".nojekyll").write_text("", encoding="utf-8")
    print(f"exported demo to {root}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="evalbench")
    sp = p.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("suite", help="build the prompt suite")
    s.add_argument("--out", default="data/suite.jsonl")
    s.set_defaults(fn=cmd_suite)
    s = sp.add_parser("run", help="generate, score, cluster and write run.json")
    s.add_argument("--config", default="configs/default.json")
    s.add_argument("--out", required=True)
    s.add_argument("--judge", help="local:<hf-id> | openai:<model> | anthropic:<model> | stub")
    s.add_argument("--only", nargs="*", help="generate for these checkpoints only, no scoring")
    s.set_defaults(fn=cmd_run)
    s = sp.add_parser("calibrate", help="compare candidate judges on recorded generations")
    s.add_argument("--config", default="configs/default.json")
    s.add_argument("--run", required=True)
    s.add_argument("--judges", nargs="+", required=True)
    s.set_defaults(fn=cmd_calibrate)
    s = sp.add_parser("report", help="print the summary of a run")
    s.add_argument("--run", required=True)
    s.set_defaults(fn=cmd_report)
    s = sp.add_parser("diff", help="what changed between two checkpoints of a run")
    s.add_argument("--run", required=True)
    s.add_argument("a")
    s.add_argument("b")
    s.set_defaults(fn=cmd_diff)
    s = sp.add_parser("targets", help="write ranked failure modes as JSONL for the next fine-tuning cycle")
    s.add_argument("--run", required=True)
    s.add_argument("--out", default="targets.jsonl")
    s.set_defaults(fn=cmd_targets)
    s = sp.add_parser("serve", help="start the API and web UI")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8101)
    s.set_defaults(fn=cmd_serve)
    s = sp.add_parser("export-demo", help="copy the web UI and a recorded run into docs/ for static hosting")
    s.add_argument("--run", required=True)
    s.add_argument("--web", default="web")
    s.add_argument("--docs", default="docs")
    s.set_defaults(fn=cmd_export)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main(sys.argv[1:])
