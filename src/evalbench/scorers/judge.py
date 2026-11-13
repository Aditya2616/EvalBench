"""Rubric judges behind one interface.

A judge answers narrow yes/no questions and returns p(yes). The local judge reads the next-token
log-probabilities of "Yes" and "No" instead of parsing free text. Hosted judges (OpenAI, Anthropic)
are optional adapters selected with EVALBENCH_JUDGE; they are unit-tested against a stubbed
transport only.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
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


def peak_rss_mb() -> float:
    """Peak resident memory of this process in MB (0.0 when the platform gives no way to ask)."""
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            class Counters(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

            c = Counters()
            c.cb = ctypes.sizeof(c)
            k32, psapi = ctypes.WinDLL("kernel32"), ctypes.WinDLL("psapi")
            k32.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
            ok = psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb)
            return round(c.PeakWorkingSetSize / 2**20, 1) if ok else 0.0
        import resource

        kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(kb / 1024 / (1024 if sys.platform == "darwin" else 1), 1)
    except Exception:
        return 0.0


class CachedJudge:
    """Wraps any judge with an on-disk cache of single answers, written after every chunk, so a stopped
    scoring run resumes where it stopped and a rerun costs nothing."""

    def __init__(self, inner: Judge, path: str | Path, chunk: int = 64):
        self.inner, self.path, self.chunk = inner, Path(path), chunk
        self.name = inner.name
        self.info = getattr(inner, "info", {})
        self.mem: dict[str, float] = {}
        self.hits = self.calls = 0
        self.store: dict[str, float] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    d = json.loads(line)
                    self.store[d["k"]] = d["p"]

    def _key(self, kind: str, prompt: str) -> str:
        return hashlib.sha1(f"{self.name}|{kind}|{prompt}".encode()).hexdigest()[:20]

    def _run(self, kind: str, prompts: list[str]) -> list[float]:
        keys = [self._key(kind, p) for p in prompts]
        todo = sorted({k: p for k, p in zip(keys, prompts, strict=True) if k not in self.store}.items(), key=lambda kp: len(kp[1]))
        self.calls += len(keys)
        self.hits += len(keys) - len(todo)
        fn = self.inner.p_yes if kind == "yes" else self.inner.p_first
        for i in range(0, len(todo), self.chunk):
            part = todo[i : i + self.chunk]
            ps = fn([p for _, p in part])
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                for (k, _), v in zip(part, ps, strict=True):
                    self.store[k] = v
                    f.write(json.dumps({"k": k, "p": v}) + "\n")
            self.info = getattr(self.inner, "info", {})
        return [self.store[k] for k in keys]

    def p_yes(self, prompts: list[str]) -> list[float]:
        return self._run("yes", prompts)

    def p_first(self, prompts: list[str]) -> list[float]:
        return self._run("first", prompts)


class LocalJudge:
    # A batch is limited by padded tokens as well as by count, so one long prompt cannot inflate memory.
    MAX_BATCH_TOKENS = 6144

    def __init__(self, model_id: str, batch_size: int | None = None, threads: int | None = None, device: str | None = None):
        import torch
        from transformers import AutoTokenizer

        from ..models import default_threads, load_causal_lm, resolve_device

        self.device, dtype = resolve_device(device)
        on_gpu = self.device.startswith("cuda")
        torch.set_num_threads(threads or (2 if on_gpu else default_threads()))
        self.torch, self.name, self.batch_size = torch, f"local:{model_id}", batch_size or (16 if on_gpu else 8)
        self.info = {"device": self.device, "dtype": dtype, "batch_size": self.batch_size}
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = load_causal_lm(model_id, self.device, dtype)

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
        texts_all = [
            self.tok.apply_chat_template(
                [{"role": "system", "content": SYSTEM}, {"role": "user", "content": p}],
                add_generation_prompt=True, tokenize=False,
            )
            for p in prompts
        ]
        lens = [len(self.tok.encode(t, add_special_tokens=False)) for t in texts_all]
        order = sorted(range(len(prompts)), key=lambda i: lens[i])  # similar lengths batch together
        res_sorted: list[float] = []
        i = 0
        while i < len(order):
            n = 1
            while n < self.batch_size and i + n < len(order) and lens[order[i + n]] * (n + 1) <= self.MAX_BATCH_TOKENS:
                n += 1
            part = order[i : i + n]
            i += n
            enc = self.tok([texts_all[j] for j in part], return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                # logits_to_keep=1: only the last position is projected onto the vocabulary. Without it the model
                # materialises batch x length x vocab logits (several GB for a 150k vocabulary).
                logits = self.model(**enc, logits_to_keep=1).logits[:, -1, :].float()
            lp = torch.log_softmax(logits, dim=-1)
            yes = torch.logsumexp(lp[:, pid], dim=-1)
            no = torch.logsumexp(lp[:, nid], dim=-1)
            res_sorted += torch.sigmoid(yes - no).tolist()
        if self.device.startswith("cuda"):
            self.info["peak_vram_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
        self.info["peak_rss_mb"] = peak_rss_mb()
        res = [0.0] * len(prompts)
        for pos_i, j in enumerate(order):
            res[j] = res_sorted[pos_i]
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
    spec = spec or os.environ.get("EVALBENCH_JUDGE") or "local:Qwen/Qwen2.5-0.5B-Instruct"
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
