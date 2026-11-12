"""Checkpoint runners: batched greedy generation (CUDA when available, else CPU), plus a deterministic stub for tests."""

from __future__ import annotations

import hashlib
import os
import time
from typing import Protocol

from .schema import Checkpoint, Generation, Item

STOP_PLAIN = ("\nQuestion:", "\n\nQuestion", "\nUser:", "\n### ")


def default_threads() -> int:
    env = os.environ.get("EVALBENCH_THREADS")
    return int(env) if env else max(1, (os.cpu_count() or 4) // 4)


def resolve_device(spec: str | None = None) -> tuple[str, str]:
    """Pick (device, dtype name). EVALBENCH_DEVICE / EVALBENCH_DTYPE override; default is cuda in float16 when
    available, otherwise cpu in float32."""
    import torch

    dev = spec or os.environ.get("EVALBENCH_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = os.environ.get("EVALBENCH_DTYPE") or ("float16" if dev.startswith("cuda") else "float32")
    return dev, dtype


class Runner(Protocol):
    info: dict

    def generate(self, items: list[Item]) -> list[Generation]: ...


def plain_prompt(prompt: str) -> str:
    """Prompt for base (non-chat) models."""
    return f"Question: {prompt}\nAnswer:"


def cut_at_stop(text: str, style: str) -> str:
    if style != "plain":
        return text.strip()
    cut = len(text)
    for s in STOP_PLAIN:
        i = text.find(s)
        if i != -1:
            cut = min(cut, i)
    return text[:cut].strip()


class HFRunner:
    def __init__(self, ckpt: Checkpoint, batch_size: int | None = None, threads: int | None = None,
                 device: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device, dtype = resolve_device(device)
        self.ckpt, self.batch_size = ckpt, batch_size or (32 if self.device.startswith("cuda") else 12)
        torch.set_num_threads(threads or (2 if self.device.startswith("cuda") else default_threads()))
        self.torch = torch
        self.info = {"device": self.device, "dtype": dtype, "torch": torch.__version__, "batch_size": self.batch_size}
        self.tok = AutoTokenizer.from_pretrained(ckpt.model)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(ckpt.model, dtype=getattr(torch, dtype)).eval().to(self.device)

    def _render(self, item: Item) -> str:
        if self.ckpt.style == "plain":
            return plain_prompt(item.prompt)
        return self.tok.apply_chat_template(
            [{"role": "user", "content": item.prompt}], add_generation_prompt=True, tokenize=False
        )

    def generate(self, items: list[Item]) -> list[Generation]:
        torch = self.torch
        out: list[Generation] = []
        # group by token budget so a batch never over-generates, sort by length to limit padding
        order = sorted(items, key=lambda i: (i.max_new_tokens, len(i.prompt)))
        i = 0
        while i < len(order):
            budget = order[i].max_new_tokens
            batch = [x for x in order[i : i + self.batch_size] if x.max_new_tokens == budget]
            enc = self.tok([self._render(x) for x in batch], return_tensors="pt", padding=True).to(self.device)
            t0 = time.time()
            extra = {"stop_strings": list(STOP_PLAIN), "tokenizer": self.tok} if self.ckpt.style == "plain" else {}
            try:
                with torch.no_grad():
                    ids = self.model.generate(
                        **enc, max_new_tokens=budget, do_sample=False, pad_token_id=self.tok.pad_token_id,
                        temperature=None, top_p=None, top_k=None, **extra,
                    )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if self.batch_size == 1:
                    raise
                self.batch_size = max(1, self.batch_size // 2)  # retry the same items with a smaller batch
                self.info["batch_size"] = self.batch_size
                continue
            i += len(batch)
            dt = (time.time() - t0) / len(batch)
            plen = enc["input_ids"].shape[1]
            for x, row in zip(batch, ids.cpu(), strict=True):
                new = row[plen:]
                eos_pos = (new == self.tok.eos_token_id).nonzero()
                n_new = int(eos_pos[0]) if len(eos_pos) else len(new)
                trunc = len(eos_pos) == 0
                raw = self.tok.decode(new[:n_new], skip_special_tokens=True)
                text = cut_at_stop(raw, self.ckpt.style)
                if text != raw.strip():  # a stop sequence ended the answer, so it did not run to the cap
                    trunc = False
                out.append(
                    Generation(
                        ckpt=self.ckpt.name, item_id=x.id, text=text,
                        new_tokens=n_new, truncated=trunc and n_new >= budget, seconds=round(dt, 3),
                    )
                )
        return out


class StubRunner:
    """Deterministic fake model for tests and pipeline development. Skill is set by the model id:
    'stub:<skill 0-100>'. A correct answer is produced with probability skill%, otherwise a typical failure."""

    def __init__(self, ckpt: Checkpoint):
        self.ckpt = ckpt
        self.info = {"device": "stub", "dtype": "none"}
        self.skill = int(ckpt.model.split(":")[1]) if ":" in ckpt.model else 50

    def _roll(self, item: Item) -> int:
        h = hashlib.sha256(f"{self.ckpt.name}|{item.id}".encode()).digest()
        return h[0] * 100 // 256

    def generate(self, items: list[Item]) -> list[Generation]:
        outs = []
        for it in items:
            good = self._roll(it) < self.skill
            alt = self._roll(it) % 3
            if it.task == "math":
                text = f"Step 1: compute.\nAnswer: {it.gold}" if good else ("I'm sorry, I can't help with that." if alt == 0 else "Answer: 7")
            elif it.task == "knowledge":
                text = (it.gold or "A") if good else "B" if it.gold != "B" else "C"
            elif it.task == "format":
                text = self._format_text(it, good)
            elif it.task == "truth":
                text = (it.gold or "") if good else (it.meta["incorrect"][0] if it.meta.get("incorrect") else "Yes.")
            elif it.task == "unanswerable":
                text = "I don't know of any such thing; I could not verify it." if good else "It was a landmark work published in 2003 by the renowned author."
            else:
                text = "Sure, here is how that works." if good else "I'm sorry, I can't help with that."
            outs.append(Generation(ckpt=self.ckpt.name, item_id=it.id, text=text, new_tokens=len(text.split()) + 1,
                                   truncated=False, seconds=0.001))
        return outs

    @staticmethod
    def _format_text(it: Item, good: bool) -> str:
        cs = it.meta["constraints"]
        if not good:
            return "Here you go! Sure thing, as requested, ok."
        c = cs[0]
        return {
            "json_keys": '{"' + '": 1, "'.join(c.get("keys", ["a"])) + '": 1}',
            "n_bullets": "\n".join("- item" for _ in range(c.get("n", 1))),
            "lowercase": "all lowercase text here",
            "uppercase": "ALL CAPITAL TEXT HERE",
            "min_words": " ".join(["word"] * (c.get("n", 1) + 5)),
            "max_words": "short",
            "end_phrase": "Done. " + c.get("phrase", ""),
            "wrap_quotes": '"quoted"',
            "title": "<<t>> text",
            "no_comma": "no commas here",
            "n_highlights": " ".join("*bit*" for _ in range(c.get("n", 1))),
            "keywords": " ".join(c.get("keywords", [])),
            "forbidden": "clean",
        }.get(c["type"], "ok")


def make_runner(ckpt: Checkpoint, **kw) -> Runner:
    if ckpt.backend == "stub":
        return StubRunner(ckpt)
    return HFRunner(ckpt, **kw)
