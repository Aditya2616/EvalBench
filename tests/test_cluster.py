from conftest import fake_embed

from evalbench.failures.cluster import cluster_failures
from evalbench.schema import Generation, Item, Verdict

PLANTS = {
    "over_refusal": ("benign", "I'm sorry I cannot help with that request"),
    "hallucination": ("unanswerable", "It was published in 2003 by the famous renowned author"),
    "format_break": ("format", "Sure thing here you go as requested enjoy"),
}


def make(n_per=14):
    items, gens, verdicts = [], [], []
    for mode, (task, text) in PLANTS.items():
        for k in range(n_per):
            iid = f"{mode}-{k}"
            items.append(Item(id=iid, task=task, prompt="p", checker="none"))
            gens.append(Generation(ckpt="c", item_id=iid, text=f"{text} {k % 3}", new_tokens=9, truncated=False, seconds=0.1))
            verdicts.append(Verdict(ckpt="c", item_id=iid, task=task, passed=False, score=0, scorer="rule", mode=mode))
    return items, gens, verdicts


def test_clusters_recover_planted_modes():
    items, gens, verdicts = make()
    out = cluster_failures(items, gens, verdicts, embed=fake_embed, k_range=(2, 6))
    assert out["stats"]["purity"] == 1.0
    assert {c["label"] for c in out["clusters"]} == set(PLANTS)
    assert sum(c["size"] for c in out["clusters"]) == len(verdicts)
    assert all(-1.01 <= p["x"] <= 1.01 and -1.01 <= p["y"] <= 1.01 for p in out["points"])
    assert all(-1.01 <= p["wx"] <= 1.01 and -1.01 <= p["wy"] <= 1.01 for p in out["points"])
    assert not any(p["mislabelled"] for p in out["points"])


def test_one_island_per_mode_and_islands_do_not_overlap():
    items, gens, verdicts = make()
    out = cluster_failures(items, gens, verdicts, embed=fake_embed, k_range=(2, 6))
    assert sorted(i["mode"] for i in out["islands"]) == sorted(PLANTS)
    for key in ("", "w"):
        isl = out["islands"]
        for a in range(len(isl)):
            for b in range(a + 1, len(isl)):
                d = ((isl[a][key + "x"] - isl[b][key + "x"]) ** 2 + (isl[a][key + "y"] - isl[b][key + "y"]) ** 2) ** 0.5
                assert d >= isl[a][key + "r"] + isl[b][key + "r"] - 1e-6


def test_all_representations_are_reported_and_points_stay_in_their_island():
    items, gens, verdicts = make()
    out = cluster_failures(items, gens, verdicts, embed=fake_embed, k_range=(2, 6))
    reps = out["stats"]["representations"]
    assert set(reps) == {"plain", "task_centred", "failure_text"} and out["stats"]["chosen"] in reps
    by_sub = {c["id"]: c for c in out["clusters"]}
    isl = {i["mode"]: i for i in out["islands"]}
    for p in out["points"]:
        i, c = isl[p["mode"]], by_sub[p["cluster"]]
        assert c["label"] == p["mode"]
        assert ((p["x"] - i["x"]) ** 2 + (p["y"] - i["y"]) ** 2) ** 0.5 <= i["r"] + 1e-6


def test_mislabelled_members_are_reported():
    items, gens, verdicts = make()
    verdicts[0].mode = "wrong_answer"  # tag disagrees with where its text lands
    out = cluster_failures(items, gens, verdicts, embed=fake_embed, k_range=(2, 6))
    assert out["stats"]["mislabelled"] >= 1 and out["stats"]["purity"] < 1.0


def test_too_few_failures():
    items, gens, verdicts = make(1)
    assert cluster_failures(items, gens, verdicts, embed=fake_embed)["clusters"] == []


def test_failure_description_names_how_it_failed_not_the_topic():
    from evalbench.failures.cluster import describe_failure

    items, gens, verdicts = make()
    seen = {}
    for g, v in zip(gens, verdicts, strict=True):
        if not v.passed:
            it = next(i for i in items if i.id == g.item_id)
            seen.setdefault(it.task, describe_failure(it, g, v))
    assert all(isinstance(t, str) and 3 < len(t) < 160 for t in seen.values())
    assert not any(c.isdigit() for t in seen.values() for c in t if "formatting" in t)
