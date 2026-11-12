import collections
from pathlib import Path

from evalbench.schema import TASKS, load_suite, suite_hash

SUITE = Path(__file__).resolve().parents[1] / "data" / "suite.jsonl"


def test_suite_loads_and_is_well_formed():
    items = load_suite(SUITE)
    assert len(items) == len({i.id for i in items}) >= 100
    assert set(collections.Counter(i.task for i in items)) == set(TASKS)
    for i in items:
        assert i.license and i.source
        if i.checker in ("exact_numeric", "exact_choice"):
            assert i.gold
        if i.checker == "constraints":
            assert i.meta["constraints"]


def test_hash_is_stable_and_content_sensitive():
    items = load_suite(SUITE)
    h = suite_hash(items)
    assert h == suite_hash(load_suite(SUITE))
    items[0].prompt += " "
    assert suite_hash(items) != h
