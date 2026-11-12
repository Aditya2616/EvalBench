import pytest

from evalbench import report
from evalbench.schema import Verdict


def v(ck, item, score, task="math", mode=None, jp=None):
    return Verdict(ckpt=ck, item_id=item, task=task, passed=score >= 1, score=score, scorer="rule", mode=mode, judge_p=jp)


def test_wilson_bounds():
    lo, hi = report.wilson(5, 10)
    assert 0.2 < lo < 0.5 < hi < 0.8
    assert report.wilson(0, 0) == (0.0, 0.0)
    assert report.wilson(10, 10)[1] == 1.0


def test_win_matrix_ties_and_symmetry():
    vs = [v("a", "1", 1), v("b", "1", 0), v("a", "2", 0), v("b", "2", 0), v("a", "3", 1), v("b", "3", 1)]
    m = report.win_matrix(vs)
    ab, ba = m["cells"]["a"]["b"], m["cells"]["b"]["a"]
    assert (ab["wins"], ab["losses"], ab["ties"]) == (1, 0, 2)
    assert ab["win_rate"] == pytest.approx(2 / 3) and ba["win_rate"] == pytest.approx(1 / 3)
    assert ab["win_rate"] + ba["win_rate"] == pytest.approx(1)


def test_field_win_rate_orders_checkpoints():
    vs = []
    for i in range(30):
        vs += [v("hi", str(i), 1), v("mid", str(i), 1 if i % 2 else 0), v("lo", str(i), 0)]
    w = report.field_win_rate(vs, boot=200)
    assert w["hi"]["win_rate"] > w["mid"]["win_rate"] > w["lo"]["win_rate"]
    assert w["hi"]["ci_lo"] <= w["hi"]["win_rate"] <= w["hi"]["ci_hi"]


def test_diff_fixed_regressed_and_modes():
    vs = [
        v("a", "1", 0, mode="wrong_answer"), v("b", "1", 1),
        v("a", "2", 1), v("b", "2", 0, mode="format_break"),
        v("a", "3", 0, mode="wrong_answer"), v("b", "3", 0, mode="wrong_answer"),
    ]
    d = report.diff(vs, "a", "b")
    assert d["fixed"] == ["1"] and d["regressed"] == ["2"] and d["still_failing"] == ["3"]
    assert d["modes_delta"]["format_break"] == 1 and d["modes_delta"]["wrong_answer"] == -1


def test_kappa_and_judge_stats():
    assert report.kappa([True, False, True, False], [True, False, True, False]) == 1.0
    assert report.kappa([True, True, False, False], [True, False, True, False]) == 0.0
    vs = [v("a", str(i), float(i % 2), jp=0.9 if i % 2 else 0.1) for i in range(10)]
    s = report.judge_stats(vs, [{"a": "a", "gold": "a", "p_a_first": 0.9, "p_a_second": 0.2}])
    assert s["strict"]["accuracy"] == 1.0 and s["strict"]["kappa"] == 1.0
    assert s["pairwise"]["flip_rate"] == 1.0


def test_targets_rank_by_count():
    vs = [v("a", "1", 0, mode="wrong_answer"), v("a", "2", 0, mode="wrong_answer"), v("a", "3", 0, mode="format_break")]
    t = report.targets(vs)
    assert t["a"][0]["mode"] == "wrong_answer" and t["a"][0]["count"] == 2


def test_cap_hits_are_counted_per_checkpoint_and_task():
    from evalbench.schema import Generation

    def g(c, i, t):
        return Generation(ckpt=c, item_id=i, text="x", new_tokens=5, truncated=t, seconds=0.1)

    out = report.cap_hits([g("a", "m1", True), g("a", "k1", False), g("b", "m1", False)], {"m1": "math", "k1": "knowledge"})
    assert out["a"]["hits"] == 1 and out["a"]["n"] == 2 and out["a"]["rate"] == 0.5
    assert out["a"]["by_task"]["math"] == {"n": 1, "hits": 1} and out["b"]["hits"] == 0
