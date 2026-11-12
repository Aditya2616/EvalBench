from evalbench.models import StubRunner, cut_at_stop, plain_prompt
from evalbench.schema import Checkpoint, Item


def test_plain_prompt_and_stop_cut():
    assert plain_prompt("2+2?") == "Question: 2+2?\nAnswer:"
    assert cut_at_stop(" 4\n\nQuestion: next one", "plain") == "4"
    assert cut_at_stop("  hi  ", "chat") == "hi"
    assert cut_at_stop("no stop here", "plain") == "no stop here"


def test_stub_runner_is_deterministic_and_skill_ordered():
    items = [Item(id=f"k{i}", task="knowledge", prompt="p", checker="exact_choice", gold="A") for i in range(80)]
    lo = StubRunner(Checkpoint(name="lo", model="stub:20", backend="stub")).generate(items)
    hi = StubRunner(Checkpoint(name="hi", model="stub:90", backend="stub")).generate(items)
    again = StubRunner(Checkpoint(name="hi", model="stub:90", backend="stub")).generate(items)
    assert [g.text for g in hi] == [g.text for g in again]
    assert sum(g.text == "A" for g in hi) > sum(g.text == "A" for g in lo)
