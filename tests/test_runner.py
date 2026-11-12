import json

from conftest import ROOT, fake_embed

from evalbench.runner import generate_all, load_config, run_all
from evalbench.schema import load_suite


def stub_cfg():
    cfg = load_config(ROOT / "configs" / "stub.json")
    cfg["suite"] = str(ROOT / "data" / "suite.jsonl")
    return cfg


def test_run_json_shape(stub_run):
    _, d = stub_run
    assert d["meta"]["n_items"] == 408 and len(d["meta"]["checkpoints"]) == 3
    assert len(d["samples"]) == 1224
    assert set(d["order"]) == set(d["items"])
    rate = {r["ckpt"]: r["rate"] for r in d["pass_rates"] if r["task"] == "ALL"}
    assert rate["stub-a"] < rate["stub-b"] < rate["stub-c"]
    assert d["clusters"]["stats"]["n_failures"] == sum(not s["p"] for s in d["samples"])
    assert d["diff"]["from"] == "stub-a" and d["diff"]["to"] == "stub-c" and d["diff"]["fixed"]
    assert d["judge"]["strict"]["n"] > 0


def test_run_is_deterministic(stub_run, tmp_path):
    d2 = run_all(stub_cfg(), tmp_path, embed=fake_embed, log=lambda s: None)
    _, d1 = stub_run
    assert d1["pass_rates"] == d2["pass_rates"] and d1["field_win"] == d2["field_win"]


def test_generation_is_resumable(tmp_path):
    cfg = stub_cfg()
    items = load_suite(cfg["suite"])
    logs: list[str] = []
    generate_all(items, cfg["checkpoints"][:1], tmp_path, logs.append)
    path = tmp_path / "generations" / "stub-a.jsonl"
    first = path.read_text().splitlines()
    assert len(first) == 408
    # simulate an interrupted run, then resume
    path.write_text("\n".join(first[:50]) + "\n")
    generate_all(items, cfg["checkpoints"][:1], tmp_path, logs.append)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 408 and len({r["item_id"] for r in rows}) == 408
    generate_all(items, cfg["checkpoints"][:1], tmp_path, logs.append)
    assert len(path.read_text().splitlines()) == 408 and "all 408 cached" in logs[-1]
