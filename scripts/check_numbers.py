#!/usr/bin/env python
"""Generate the numbers in README.md from the recorded run, or check that the README is in sync.

    python scripts/check_numbers.py --write    # rewrite every generated block and inline number
    python scripts/check_numbers.py --check    # exit 1 if the README, docs/data/run.json or suite metadata disagree

Generated blocks sit between `<!-- numbers:NAME -->` and `<!-- /numbers:NAME -->`. Single numbers in prose sit
between `<!-- n:NAME -->` and `<!-- /n:NAME -->`. Nothing inside those markers is typed by hand. The same run
file is what the web UI and the static demo read (`docs/data/run.json` must be byte-identical to `runs/main/run.json`).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASKS = ["math", "knowledge", "format", "truth", "unanswerable", "benign"]
MODE_NAMES = {"hallucination": "hallucination", "format_break": "format break", "wrong_answer": "wrong answer",
              "over_refusal": "over-refusal", "degenerate": "degenerate"}


def pct(x: float, d: int = 1) -> str:
    return f"{x * 100:.{d}f}%"


def load(run_dir: Path) -> dict:
    return json.loads((run_dir / "run.json").read_text(encoding="utf-8"))


def rates(d: dict) -> dict[tuple[str, str], dict]:
    return {(r["ckpt"], r["task"]): r for r in d["pass_rates"]}


def names(d: dict) -> list[str]:
    return [c["name"] for c in d["meta"]["checkpoints"]]


# ------------------------------------------------------------------ blocks

def b_setup(d: dict) -> str:
    m = d["meta"]
    dev = sorted({(v.get("device"), v.get("dtype")) for k, v in m["devices"].items() if not k.startswith("judge:")})
    dev_txt = ", ".join(f"{a} in {b}" for a, b in dev)
    judge_dev = sorted({(v.get("device"), v.get("dtype")) for k, v in m["devices"].items() if k == "judge:" + m["judge"]})
    jtxt = ", ".join(f"{a} in {b}" for a, b in judge_dev) or "unknown"
    return (f"Run recorded on {m['created']} from commit `{m['commit']}`: suite version {m['suite_version']} "
            f"(hash `{m['suite_hash']}`), {m['n_items']} prompts, {len(m['checkpoints'])} checkpoints "
            f"({', '.join(names(d))}), generation on {dev_txt}, default judge `{m['judge']}` on {jtxt}.")


def b_passtable(d: dict) -> str:
    R = rates(d)
    ns = {t: R[(names(d)[0], t)]["n"] for t in TASKS}
    head = "| checkpoint | " + " | ".join(f"{t} (n={ns[t]})" for t in TASKS) + " | all (95% Wilson interval) |"
    rows = [head, "|" + "---|" * (len(TASKS) + 2)]
    for c in names(d):
        cells = [f"{R[(c, t)]['passed']}/{R[(c, t)]['n']}" for t in TASKS]
        a = R[(c, "ALL")]
        rows.append(f"| {c} | " + " | ".join(cells) + f" | {a['passed']}/{a['n']}, {pct(a['rate'])} [{pct(a['ci_lo'])}, {pct(a['ci_hi'])}] |")
    return "\n".join(rows)


def b_interval(d: dict) -> str:
    R = rates(d)
    widths = [(R[(c, 'ALL')]['ci_hi'] - R[(c, 'ALL')]['ci_lo']) / 2 * 100 for c in names(d)]
    return f"half-width of the 95% intervals: {min(widths):.1f} to {max(widths):.1f} points"


def b_winrate(d: dict) -> str:
    fw = d["field_win"]
    order = sorted(names(d), key=lambda c: -fw[c]["win_rate"])
    return ", ".join(f"{c} {fw[c]['win_rate']:.3f} [{fw[c]['ci_lo']:.3f}, {fw[c]['ci_hi']:.3f}]" for c in order)


def b_caps(d: dict) -> str:
    ch = d["meta"]["cap_hits"]
    rows = ["| checkpoint | ran to the token cap | of which loops (more tokens would not help) | budget only | mean seconds per prompt |",
            "|---|---|---|---|---|"]
    for c in names(d):
        h = ch[c]
        rows.append(f"| {c} | {h['hits']}/{h['n']} ({pct(h['rate'])}) | {h['loops']} | {h['budget_only']} ({pct(h['budget_rate'])}) "
                    f"| {d['meta']['seconds_per_item'][c]:.2f} |")
    return "\n".join(rows)


def b_degen(d: dict) -> str:
    """Truly degenerate output (empty, looping, gibberish) per checkpoint, apart from cap hits."""
    mt = d["mode_table"]
    rows = ["| checkpoint | " + " | ".join(MODE_NAMES.values()) + " | total failures |", "|" + "---|" * (len(MODE_NAMES) + 2)]
    for c in names(d):
        rows.append(f"| {c} | " + " | ".join(str(mt[c][m]) for m in MODE_NAMES) + f" | {sum(mt[c].values())} |")
    return "\n".join(rows)


def b_diff(d: dict) -> str:
    x = d["diff"]
    n = x["n"]
    pre = (f"{x['from']} to {x['to']}, same {n} prompts, counted per prompt: {len(x['fixed'])} fixed (failed before, pass now), "
           f"{len(x['regressed'])} regressed (passed before, fail now), {len(x['still_failing'])} still failing, "
           f"{n - len(x['fixed']) - len(x['regressed']) - len(x['still_failing'])} passed both times.")
    rows = ["| failure mode | " + x["from"] + " | " + x["to"] + " | change |", "|---|---|---|---|"]
    for m, label in MODE_NAMES.items():
        rows.append(f"| {label} | {x['modes_from'][m]} | {x['modes_to'][m]} | {x['modes_delta'][m]:+d} |")
    rows.append(f"| all failures | {sum(x['modes_from'].values())} | {sum(x['modes_to'].values())} | "
                f"{sum(x['modes_to'].values()) - sum(x['modes_from'].values()):+d} |")
    return pre + "\n\n" + "\n".join(rows)


def b_clusters(d: dict) -> str:
    st = d["clusters"]["stats"]
    rows = ["| representation | k | silhouette | purity vs detector mode | AMI vs mode | AMI vs task | samples whose mode disagrees with their cluster |",
            "|---|---|---|---|---|---|---|"]
    label = {"plain": "plain (\"[task] output\" embedding)", "task_centred": "task-centred (output embedding minus its task mean)"}
    for k, r in st["representations"].items():
        mark = " (used for the map)" if k == st["chosen"] else ""
        rows.append(f"| {label.get(k, k)}{mark} | {r['k']} | {r['silhouette']:.3f} | {r['purity']:.3f} | {r['ami_vs_mode']:.3f} | "
                    f"{r['ami_vs_task']:.3f} | {r['mislabelled']} of {st['n_failures']} |")
    return "\n".join(rows)


def b_islands(d: dict) -> str:
    cl = {c["id"]: c for c in d["clusters"]["clusters"]}
    out = []
    for i in d["clusters"]["islands"]:
        subs = "; ".join(f"{cl[s]['name']} ({cl[s]['size']})" for s in i["subs"])
        out.append(f"- **{MODE_NAMES.get(i['mode'], i['mode'])}**, {i['n']} failures, {i['disagree']} with a disagreeing embedding cluster: {subs}")
    return "\n".join(out)


def b_judge(d: dict) -> str:
    jc = d.get("judge_comparison")
    if not jc:
        return "(no judge comparison recorded)"
    rows = ["| judge | strict accuracy (n) | Cohen's kappa | lexicon agreement (n) | pairwise accuracy, swap-averaged (n) | flip rate |",
            "|---|---|---|---|---|---|"]
    for spec, r in jc["judges"].items():
        s, w, p = r["strict"], r["lexicon_agreement"], r["pairwise"]
        mark = " (default)" if spec == jc["best_by_strict_kappa"] else ""
        rows.append(f"| {spec.split(':', 1)[1]}{mark} | {pct(s['accuracy'])} ({s['n']}) | {s['kappa']:.2f} | {pct(w['accuracy'], 0)}, kappa {w['kappa']:.2f} ({w['n']}) | "
                    f"{pct(p['accuracy_swap_averaged'], 0)} ({p['n']}) | {pct(p['flip_rate'], 0)} |")
    return "\n".join(rows)


def b_judgeyes(d: dict) -> str:
    jc = d["judge_comparison"]
    return "; ".join(f"{s.split(':', 1)[1]} says yes to {pct(r['strict']['judge_yes_rate'], 0)} when {pct(r['strict']['gold_pass_rate'], 0)} are correct"
                     for s, r in jc["judges"].items())


def b_memory(d: dict) -> str:
    out = []
    for k, v in d["meta"]["devices"].items():
        if k.startswith("judge:") and "peak_rss_mb" in v:
            out.append(f"{k[6:]}: peak RAM {v['peak_rss_mb'] / 1024:.1f} GB" + (f", peak VRAM {v['peak_vram_mb'] / 1024:.1f} GB" if "peak_vram_mb" in v else ""))
    return "; ".join(out) or "not recorded"


BLOCKS = {"setup": b_setup, "passtable": b_passtable, "interval": b_interval, "winrate": b_winrate, "caps": b_caps,
          "modes": b_degen, "diff": b_diff, "clusters": b_clusters, "islands": b_islands, "judge": b_judge,
          }


def sign_test_p(a: int, b: int) -> float:
    """Two-sided exact binomial (sign test) p-value for a discordant split a:b under a 50:50 null."""
    from math import comb

    n, k = a + b, min(a, b)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)


def inline(d: dict) -> dict[str, str]:
    m, st, x = d["meta"], d["clusters"]["stats"], d["diff"]
    R = rates(d)
    return {
        "n_items": str(m["n_items"]), "n_checkpoints": str(len(m["checkpoints"])), "n_samples": str(len(d["samples"])),
        "suite_version": m["suite_version"], "suite_hash": m["suite_hash"], "commit": m["commit"],
        "n_failures": str(st["n_failures"]), "n_islands": str(st["n_islands"]), "n_sub": str(st["n_sub"]),
        "diff_fixed": str(len(x["fixed"])), "diff_regressed": str(len(x["regressed"])), "diff_still": str(len(x["still_failing"])),
        "diff_from": x["from"], "diff_to": x["to"],
        "diff_p": f"{sign_test_p(len(x['fixed']), len(x['regressed'])):.4f}",
        "diff_n": str(x["n"]),
        "gap_it_base": f"{(R[(x['to'], 'ALL')]['rate'] - R[(x['from'], 'ALL')]['rate']) * 100:.1f}",
        "base_hall": str(d["mode_table"][x["from"]]["hallucination"]), "it_hall": str(d["mode_table"][x["to"]]["hallucination"]),
        "it_degen": str(d["mode_table"][x["to"]]["degenerate"]), "base_degen": str(d["mode_table"][x["from"]]["degenerate"]),
        "it_loops": str(m["cap_hits"][x["to"]]["loops"]), "it_budget": str(m["cap_hits"][x["to"]]["budget_only"]),
        "half_lo": f"{min((R[(c, 'ALL')]['ci_hi'] - R[(c, 'ALL')]['ci_lo']) / 2 * 100 for c in names(d)):.1f}",
        "half_hi": f"{max((R[(c, 'ALL')]['ci_hi'] - R[(c, 'ALL')]['ci_lo']) / 2 * 100 for c in names(d)):.1f}",
        "n_disagree_plain": str(st["representations"]["plain"]["mislabelled"]),
        "degen_disagree": str(next(i["disagree"] for i in d["clusters"]["islands"] if i["mode"] == "degenerate")),
        "degen_n": str(next(i["n"] for i in d["clusters"]["islands"] if i["mode"] == "degenerate")),
        "plain_sil": f"{st['representations']['plain']['silhouette']:.3f}", "plain_purity": f"{st['representations']['plain']['purity']:.3f}",
        "centred_sil": f"{st['representations']['task_centred']['silhouette']:.3f}",
        "centred_purity": f"{st['representations']['task_centred']['purity']:.3f}",
        "chosen_mislabelled": str(st["mislabelled"]),
        "n_math": str(R[(names(d)[0], "math")]["n"]),
        "default_judge": m["judge"].split(":", 1)[1], "n_strict": str(d["judge_comparison"]["judges"][m["judge"]]["strict"]["n"]),
        "n_pairs": str(d["judge_comparison"]["judges"][m["judge"]]["pairwise"]["n"]),
        "judgeyes": b_judgeyes(d), "memory": b_memory(d),
    }


# ------------------------------------------------------------------ README rewriting

def render(text: str, d: dict) -> str:
    def block(m: re.Match) -> str:
        name = m.group(1)
        if name not in BLOCKS:
            raise SystemExit(f"README has an unknown block 'numbers:{name}'")
        return f"<!-- numbers:{name} -->\n{BLOCKS[name](d)}\n<!-- /numbers:{name} -->"

    text = re.sub(r"<!-- numbers:(\w+) -->\n.*?\n<!-- /numbers:\1 -->", block, text, flags=re.S)
    vals = inline(d)

    def one(m: re.Match) -> str:
        if m.group(1) not in vals:
            raise SystemExit(f"README has an unknown inline number 'n:{m.group(1)}'")
        return f"<!-- n:{m.group(1)} -->{vals[m.group(1)]}<!-- /n:{m.group(1)} -->"

    return re.sub(r"<!-- n:(\w+) -->.*?<!-- /n:\1 -->", one, text)


def problems(root: Path = ROOT, run: str = "runs/main") -> list[str]:
    """Everything that is out of sync, as readable lines. Empty means in sync."""
    out: list[str] = []
    run_dir = root / run
    d = load(run_dir)
    readme = (root / "README.md").read_text(encoding="utf-8")
    if render(readme, d) != readme:
        out.append("README.md numbers are out of sync with the run (run: python scripts/check_numbers.py --write)")
    demo = root / "docs" / "data" / "run.json"
    if not demo.exists() or demo.read_bytes() != (run_dir / "run.json").read_bytes():
        out.append("docs/data/run.json is not identical to runs/main/run.json (run: evalbench export-demo --run runs/main)")
    meta = json.loads((root / "data" / "suite.meta.json").read_text(encoding="utf-8"))
    m = d["meta"]
    if (meta["hash"], meta["n"], meta["version"]) != (m["suite_hash"], m["n_items"], m["suite_version"]):
        out.append("data/suite.meta.json disagrees with the run's suite hash, size or version")
    if sum(meta["per_task"].values()) != meta["n"]:
        out.append("suite.meta.json per-task counts do not add up to n")
    R = rates(d)
    for c in names(d):
        if sum(R[(c, t)]["n"] for t in TASKS) != R[(c, "ALL")]["n"] or sum(R[(c, t)]["passed"] for t in TASKS) != R[(c, "ALL")]["passed"]:
            out.append(f"{c}: per-task counts do not add up to the total")
        if sum(d["mode_table"][c].values()) != R[(c, "ALL")]["n"] - R[(c, "ALL")]["passed"]:
            out.append(f"{c}: failure modes do not add up to the failures")
    for f in ("app.js", "index.html", "style.css"):
        if (root / "web" / f).read_bytes() != (root / "docs" / f).read_bytes():
            out.append(f"docs/{f} differs from web/{f} (run: evalbench export-demo --run runs/main)")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true")
    g.add_argument("--check", action="store_true")
    p.add_argument("--run", default="runs/main")
    a = p.parse_args()
    d = load(ROOT / a.run)
    path = ROOT / "README.md"
    text = path.read_text(encoding="utf-8")
    if a.write:
        new = render(text, d)
        path.write_text(new, encoding="utf-8")
        print("README.md updated" if new != text else "README.md already in sync")
        return
    bad = problems(ROOT, a.run)
    for line in bad:
        print("OUT OF SYNC:", line)
    if bad:
        sys.exit(1)
    print("numbers in sync")


if __name__ == "__main__":
    main()
