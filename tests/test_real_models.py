import pytest

from evalbench.models import HFRunner
from evalbench.schema import Checkpoint, load_suite
from evalbench.scorers.judge import LocalJudge, rubric_prompt

pytestmark = pytest.mark.slow


def test_smollm_generates_and_local_judge_returns_probabilities():
    items = [i for i in load_suite("data/suite.jsonl") if i.task == "knowledge"][:3]
    ck = Checkpoint(name="t", model="HuggingFaceTB/SmolLM2-135M-Instruct")
    runner = HFRunner(ck, batch_size=3, threads=2)
    gens = runner.generate(items)
    assert runner.info["device"] in ("cuda", "cpu") and runner.info["dtype"] in ("float16", "float32")
    assert len(gens) == 3 and all(g.new_tokens <= i.max_new_tokens for g, i in zip(gens, items, strict=True))
    j = LocalJudge("HuggingFaceTB/SmolLM2-135M-Instruct", threads=2)
    ps = j.p_yes([rubric_prompt("Does the response say hello?", "Greet me", "Hello there!")])
    assert 0.0 <= ps[0] <= 1.0
    assert 0.0 <= j.p_first(["Which is better? Answer A or B."])[0] <= 1.0
