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
    assert not any(p["mislabelled"] for p in out["points"])


def test_mislabelled_members_are_reported():
    items, gens, verdicts = make()
    verdicts[0].mode = "wrong_answer"  # tag disagrees with where its text lands
    out = cluster_failures(items, gens, verdicts, embed=fake_embed, k_range=(2, 6))
    assert out["stats"]["mislabelled"] >= 1 and out["stats"]["purity"] < 1.0


def test_too_few_failures():
    items, gens, verdicts = make(1)
    assert cluster_failures(items, gens, verdicts, embed=fake_embed)["clusters"] == []
