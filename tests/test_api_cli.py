import json

import pytest
from conftest import ROOT
from fastapi.testclient import TestClient

from evalbench import cli
from evalbench.api import create_app


@pytest.fixture()
def client(stub_run, tmp_path):
    run_dir, _ = stub_run
    runs = tmp_path / "runs"
    (runs / "r1").mkdir(parents=True)
    (runs / "r1" / "run.json").write_bytes((run_dir / "run.json").read_bytes())
    return TestClient(create_app(runs_dir=runs, config_path=ROOT / "configs" / "stub.json", web_dir=tmp_path / "none"))


def test_health_and_runs(client):
    assert client.get("/health").json() == {"status": "ok", "version": "0.1.0", "runs": 1}
    runs = client.get("/api/runs").json()
    assert runs[0]["id"] == "r1" and runs[0]["checkpoints"] == ["stub-a", "stub-b", "stub-c"]


def test_samples_filters(client):
    all_f = client.get("/api/runs/r1/samples?limit=500").json()
    assert all_f and all(not s["passed"] for s in all_f)
    only = client.get("/api/runs/r1/samples?ckpt=stub-a&task=benign&limit=500").json()
    assert only and all(s["ckpt"] == "stub-a" and s["task"] == "benign" for s in only)
    run = client.get("/api/runs/r1").json()
    cl = run["clusters"]["clusters"][0]
    members = client.get(f"/api/runs/r1/samples?cluster={cl['id']}&limit=500").json()
    assert len(members) == cl["size"]


def test_unknown_run_and_latest(client):
    assert client.get("/api/runs/nope").status_code == 404
    assert client.get("/data/run.json").json()["meta"]["n_items"] == 120


def test_try_prompt_with_stub_models(client):
    r = client.post("/api/try", json={"prompt": "What is 2+2?", "checkpoints": ["stub-a", "stub-c"]})
    assert r.status_code == 200 and len(r.json()) == 2 and "valid_json" in r.json()[0]["signals"]
    assert client.post("/api/try", json={"prompt": "x", "checkpoints": ["nope"]}).status_code == 422
    assert client.post("/api/try", json={"prompt": "", "checkpoints": ["stub-a"]}).status_code == 422


def test_cli_report_diff_targets_export(stub_run, tmp_path, capsys):
    run_dir, _ = stub_run
    cli.main(["report", "--run", str(run_dir)])
    assert "win-rate against the field" in capsys.readouterr().out
    cli.main(["diff", "--run", str(run_dir), "stub-a", "stub-c"])
    assert "fixed" in capsys.readouterr().out
    out = tmp_path / "t.jsonl"
    cli.main(["targets", "--run", str(run_dir), "--out", str(out)])
    assert json.loads(out.read_text().splitlines()[0])["checkpoint"]
    web = tmp_path / "web"
    web.mkdir()
    (web / "index.html").write_text("<html></html>")
    cli.main(["export-demo", "--run", str(run_dir), "--web", str(web), "--docs", str(tmp_path / "docs")])
    assert (tmp_path / "docs" / "data" / "run.json").exists() and (tmp_path / "docs" / "index.html").exists()
    assert json.loads((tmp_path / "docs" / "data" / "config.json").read_text()) == {"try_enabled": False}
