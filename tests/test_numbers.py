import importlib.util
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("check_numbers", ROOT / "scripts" / "check_numbers.py")
cn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cn)


def test_readme_and_demo_are_in_sync_with_the_recorded_run():
    assert cn.problems(ROOT) == []


def test_render_is_idempotent_and_detects_edits():
    d = cn.load(ROOT / "runs" / "main")
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    once = cn.render(text, d)
    assert cn.render(once, d) == once
    first = d["meta"]["n_items"]
    tampered = text.replace(f"<!-- n:n_items -->{first}<!-- /n:n_items -->", "<!-- n:n_items -->1<!-- /n:n_items -->", 1)
    assert tampered != text and cn.render(tampered, d) == text


def test_problems_flags_a_stale_demo_file(tmp_path):
    for name in ("README.md",):
        shutil.copy2(ROOT / name, tmp_path / name)
    for sub in ("runs/main", "docs/data", "data", "web", "docs"):
        (tmp_path / sub).mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "runs/main/run.json", tmp_path / "runs/main/run.json")
    shutil.copy2(ROOT / "data/suite.meta.json", tmp_path / "data/suite.meta.json")
    for f in ("app.js", "index.html", "style.css"):
        shutil.copy2(ROOT / "web" / f, tmp_path / "web" / f)
        shutil.copy2(ROOT / "web" / f, tmp_path / "docs" / f)
    shutil.copy2(ROOT / "runs/main/run.json", tmp_path / "docs/data/run.json")
    assert cn.problems(tmp_path) == []
    data = json.loads((tmp_path / "docs/data/run.json").read_text(encoding="utf-8"))
    data["meta"]["n_items"] += 1
    (tmp_path / "docs/data/run.json").write_text(json.dumps(data), encoding="utf-8")
    assert any("run.json" in p for p in cn.problems(tmp_path))


def test_claims_made_in_the_readme_prose_hold_for_the_recorded_run():
    d = cn.load(ROOT / "runs" / "main")
    R = cn.rates(d)
    q, it, base, small = "qwen2.5-0.5b-it", "smollm2-360m-it", "smollm2-360m-base", "smollm2-135m-it"
    top = {c: R[(c, "ALL")] for c in (q, it, base, small)}
    assert top[q]["ci_lo"] > top[small]["ci_hi"] and top[q]["ci_lo"] > top[base]["ci_hi"]  # Qwen clear of 135M and base
    assert abs(top[q]["ci_lo"] - top[it]["ci_hi"]) < 0.005  # and only just touches 360M Instruct
    assert top[small]["ci_lo"] < top[base]["ci_hi"] and top[base]["ci_lo"] < top[small]["ci_hi"]  # 135M and base overlap
    assert all(R[(c, "math")]["passed"] <= 10 for c in (small, base, it))  # SmolLM2 solves very few maths items
    assert all(R[(c, "unanswerable")]["passed"] <= 3 for c in (small, base, it))
    assert R[(base, "benign")]["passed"] == R[(base, "benign")]["n"] and all(R[(c, "benign")]["passed"] < R[(c, "benign")]["n"] for c in (small, it, q))
    x = d["diff"]
    assert len(x["fixed"]) > len(x["regressed"]) and cn.sign_test_p(len(x["fixed"]), len(x["regressed"])) < 0.01
    assert x["modes_delta"]["format_break"] < 0 and x["modes_delta"]["wrong_answer"] < 0 and x["modes_delta"]["degenerate"] > 0
    assert d["meta"]["cap_hits"][it]["loops"] > d["meta"]["cap_hits"][it]["budget_only"]
    assert d["judge_comparison"]["best_by_strict_kappa"] == d["meta"]["judge"]
