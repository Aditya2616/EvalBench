import json
import math

import httpx
import pytest

from evalbench.scorers.judge import (
    AnthropicJudge,
    OpenAIJudge,
    StubJudge,
    make_judge,
    pair_prompt,
    rubric_prompt,
)


def openai_transport(token_probs):
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path.endswith("/chat/completions")
        assert req.headers["authorization"] == "Bearer k"
        body = json.loads(req.content)
        assert body["logprobs"] is True and body["max_tokens"] == 1
        top = [{"token": t, "logprob": math.log(p)} for t, p in token_probs]
        return httpx.Response(200, json={"choices": [{"logprobs": {"content": [{"top_logprobs": top}]}}]})

    return httpx.MockTransport(handler)


def test_openai_judge_uses_logprobs_over_stub_transport():
    j = OpenAIJudge("m", api_key="k", base_url="http://x/v1",
                    client=httpx.Client(transport=openai_transport([("Yes", 0.6), (" yes", 0.1), ("No", 0.3)])))
    assert j.p_yes([rubric_prompt("q", "p", "r")])[0] == pytest.approx(0.7)
    j2 = OpenAIJudge("m", api_key="k", base_url="http://x/v1",
                     client=httpx.Client(transport=openai_transport([("A", 0.2), ("B", 0.8)])))
    assert j2.p_first([pair_prompt("p", "a", "b")])[0] == pytest.approx(0.2)


def test_anthropic_judge_over_stub_transport():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["h"] = dict(req.headers)
        text = "Yes" if "needle" in req.content.decode() else "No."
        return httpx.Response(200, json={"content": [{"type": "text", "text": text}]})

    j = AnthropicJudge("m", api_key="k", base_url="http://x", client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert j.p_yes([rubric_prompt("needle?", "p", "r"), rubric_prompt("q", "p", "r")]) == [1.0, 0.0]
    assert seen["h"]["x-api-key"] == "k" and seen["h"]["anthropic-version"]


def test_hosted_judge_http_error_propagates():
    j = OpenAIJudge("m", api_key="k", base_url="http://x/v1",
                    client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(401, json={}))))
    with pytest.raises(httpx.HTTPStatusError):
        j.p_yes(["x"])


def test_make_judge_selection(monkeypatch):
    assert isinstance(make_judge("stub"), StubJudge)
    assert make_judge("openai:gpt-x", client=httpx.Client()).name == "openai:gpt-x"
    assert make_judge("anthropic:test-model", client=httpx.Client()).name == "anthropic:test-model"
    monkeypatch.setenv("EVALBENCH_JUDGE", "stub")
    assert make_judge().name == "stub"
    with pytest.raises(ValueError):
        make_judge("nope:x")


def test_stub_judge_reads_reference():
    q = "Reference final answer: 18. Does the response arrive at this same final answer?"
    j = StubJudge()
    assert j.p_yes([rubric_prompt(q, "p", "so 18")])[0] > 0.5
    assert j.p_yes([rubric_prompt(q, "p", "so 19")])[0] < 0.5
