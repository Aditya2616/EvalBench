"""Rubric judges behind one interface.

A judge answers narrow yes/no questions and returns p(yes). The local judge reads the next-token
log-probabilities of "Yes" and "No" instead of parsing free text. Hosted judges (OpenAI, Anthropic)
are optional adapters selected with EVALBENCH_JUDGE; they are unit-tested against a stubbed
transport only.
"""

from __future__ import annotations

import math
import os
import re
from typing import Protocol

import httpx

SYSTEM = "You are a strict, literal grader. Reply with a single word."


def rubric_prompt(question: str, prompt: str, response: str) -> str:
    return (
        f"Task given to a model:\n{prompt.strip()[:900]}\n\n"
        f"Model response:\n{response.strip()[:900] or '(empty)'}\n\n"
        f"Grading question: {question}\nAnswer Yes or No."
    )


def pair_prompt(prompt: str, a: str, b: str) -> str:
    return (
        f"Task given to two models:\n{prompt.strip()[:700]}\n\n"
        f"Response A:\n{a.strip()[:600] or '(empty)'}\n\nResponse B:\n{b.strip()[:600] or '(empty)'}\n\n"
        "Which response completes the task more correctly? Answer A or B."
    )


class Judge(Protocol):
    name: str

    def p_yes(self, prompts: list[str]) -> list[float]: ...

    def p_first(self, prompts: list[str]) -> list[float]:
        """p(response A is better) for prompts built with pair_prompt."""
        ...


class LocalJudge:
    def __init__(self, model_id: str, batch_size: int = 8, threads: int | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from ..models import default_threads

        torch.set_num_threads(threads or default_threads())
        self.torch, self.name, self.batch_size = torch, f"local:{model_id}", batch_size
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32).eval()

    def _ids(self, words: list[str]) -> list[int]:
        ids = set()
        for w in words:
            for v in (w, " " + w):
                t = self.tok.encode(v, add_special_tokens=False)
                if len(t) == 1:
                    ids.add(t[0])
        return sorted(ids)

    def _two_way(self, prompts: list[str], pos: list[str], neg: list[str]) -> list[float]:
        torch = self.torch
        pid, nid = self._ids(pos), self._ids(neg)
        res: list[float] = []
        for i in range(0, len(prompts), self.batch_size):
            texts = [
                self.tok.apply_chat_template(
                    [{"role": "system", "content": SYSTEM}, {"role": "user", "content": p}],
                    add_generation_prompt=True, tokenize=False,
                )
                for p in prompts[i : i + self.batch_size]
            ]
            enc = self.tok(texts, return_tensors="pt", padding=True)
            with torch.no_grad():
                logits = self.model(**enc).logits[:, -1, :].float()
            lp = torch.log_softmax(logits, dim=-1)
            yes = torch.logsumexp(lp[:, pid], dim=-1)
            no = torch.logsumexp(lp[:, nid], dim=-1)
            res += torch.sigmoid(yes - no).tolist()
        return res

    def p_yes(self, prompts: list[str]) -> list[float]:
        return self._two_way(prompts, ["Yes", "yes", "YES"], ["No", "no", "NO"])

    def p_first(self, prompts: list[str]) -> list[float]:
        return self._two_way(prompts, ["A"], ["B"])


class _Hosted:
    name = "hosted"

    def __init__(self, client: httpx.Client | None = None):
        self.client = client or httpx.Client(timeout=60)

    def _complete(self, user: str) -> float:  # pragma: no cover - overridden
        raise NotImplementedError

    def p_yes(self, prompts: list[str]) -> list[float]:
        return [self._complete(p) for p in prompts]

    def p_first(self, prompts: list[str]) -> list[float]:
        return [self._complete(p, first=True) for p in prompts]


class OpenAIJudge(_Hosted):
    """Chat Completions with token logprobs, so the answer is a probability like the local judge."""

    def __init__(self, model: str, api_key: str | None = None, base_url: str | None = None, client=None):
        super().__init__(client)
        self.model, self.name = model, f"openai:{model}"
        self.key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base = (base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")

    def _complete(self, user: str, first: bool = False) -> float:
        r = self.client.post(
            f"{self.base}/chat/completions",
            headers={"Authorization": f"Bearer {self.key}"},
            json={
                "model": self.model, "max_tokens": 1, "temperature": 0, "logprobs": True, "top_logprobs": 10,
                "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
            },
        )
        r.raise_for_status()
        top = r.json()["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        pos, neg = ("a", "b") if first else ("yes", "no")
        p = sum(math.exp(t["logprob"]) for t in top if t["token"].strip().lower() == pos)
        n = sum(math.exp(t["logprob"]) for t in top if t["token"].strip().lower() == neg)
        return p / (p + n) if p + n else 0.5


class AnthropicJudge(_Hosted):
    """Messages API. No token logprobs are exposed, so the result is a hard 0/1 (0.5 if unparseable)."""

    def __init__(self, model: str, api_key: str | None = None, base_url: str | None = None, client=None):
        super().__init__(client)
        self.model, self.name = model, f"anthropic:{model}"
        self.key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.base = (base_url or os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")).rstrip("/")

    def _complete(self, user: str, first: bool = False) -> float:
        r = self.client.post(
            f"{self.base}/v1/messages",
            headers={"x-api-key": self.key, "anthropic-version": "2023-06-01"},
            json={"model": self.model, "max_tokens": 4, "system": SYSTEM, "messages": [{"role": "user", "content": user}]},
        )
        r.raise_for_status()
        text = "".join(b.get("text", "") for b in r.json()["content"]).strip().lower()
        pos, neg = ("a", "b") if first else ("yes", "no")
        if re.match(rf"^{pos}\b", text):
            return 1.0
        if re.match(rf"^{neg}\b", text):
            return 0.0
        return 0.5


class StubJudge:
    """Offline stand-in for tests: says yes when the reference answer appears in the response."""

    name = "stub"

    def p_yes(self, prompts: list[str]) -> list[float]:
        out = []
        for p in prompts:
            m = re.search(r"Reference (?:final )?answer: (?:option )?(.+?)\.", p)
            resp = p.split("Model response:\n", 1)[-1].split("\n\nGrading question", 1)[0]
            if m:
                out.append(0.9 if m.group(1).strip().lower() in resp.lower() else 0.1)
            else:
                from .lexicon import abstain_hit, refusal_hit

                q = p.rsplit("Grading question:", 1)[-1]
                hit = abstain_hit(resp) if "admit" in q else refusal_hit(resp)
                out.append(0.9 if hit else 0.1)
        return out

    def p_first(self, prompts: list[str]) -> list[float]:
        return [0.5 for _ in prompts]


def make_judge(spec: str | None = None, client: httpx.Client | None = None) -> Judge:
    """spec: 'local:<hf-id>' | 'openai:<model>' | 'anthropic:<model>' | 'stub'. Defaults to EVALBENCH_JUDGE."""
    spec = spec or os.environ.get("EVALBENCH_JUDGE") or "local:Qwen/Qwen2.5-1.5B-Instruct"
    kind, _, rest = spec.partition(":")
    if kind == "stub":
        return StubJudge()
    if kind == "openai":
        return OpenAIJudge(rest, client=client)
    if kind == "anthropic":
        return AnthropicJudge(rest, client=client)
    if kind == "local":
        return LocalJudge(rest)
    raise ValueError(f"unknown judge spec {spec!r}")
