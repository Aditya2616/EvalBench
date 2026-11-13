"""Embed failing samples, cluster them, and lay them out as one island per failure mode.

The failure *mode* of a sample (hallucination, format break, ...) comes from the deterministic
detectors in `scorers.score`, never from a model. The embedding is only used for two things:
(1) an unsupervised clustering that is compared with the detector tags (purity, silhouette,
adjusted mutual information) so the agreement is a measured number, and (2) sub-clusters inside
each mode island, named by what they contain. Every sample whose detector mode disagrees with
the label of its embedding cluster is flagged, and a raw t-SNE projection is exported as well so
nothing is hidden behind the island layout.

Three representations are compared and all three are reported:
  plain            embedding of "[task] output", the unsupervised baseline (it mostly groups by topic);
  task_centred     the output embedding minus the mean output embedding of its task, which removes most of
                   "what the question was about";
  failure_text     an embedding of a short description of how the answer failed, written from the verdict
                   (the scorer's evidence, whether it ran to the cap or looped), plus the task-centred output
                   embedding. Because the description is built from the detectors' evidence, its agreement with
                   the detector modes is partly by construction: it is a map of failure descriptions, not an
                   independent test of the detectors. The plain baseline is the independent one.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_mutual_info_score, silhouette_score

from ..schema import Generation, Item, Verdict

EmbedFn = Callable[[list[str]], np.ndarray]
_MODEL = None
WIDE_ASPECT = 2.3


def minilm_embed(texts: list[str]) -> np.ndarray:
    global _MODEL
    if _MODEL is None:
        import torch
        from sentence_transformers import SentenceTransformer

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=dev)
    return np.asarray(_MODEL.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False))


def failure_text(item: Item, gen: Generation) -> str:
    return f"[{item.task}] {gen.text.strip()[:400]}"


def describe_failure(item: Item, gen: Generation, v: Verdict) -> str:
    """One short sentence on how this answer failed, from the verdict evidence (no model involved)."""
    ev, task = v.evidence, item.task
    if not gen.text.strip():
        how = "empty output"
    elif task == "math":
        how = "no final answer line" if "extracted None" in ev else "wrong final number"
    elif task == "knowledge":
        how = "no option letter given" if "extracted None" in ev else "picked a wrong option"
    elif task == "format":
        how = "broke a formatting rule: " + re.sub(r"\[.*?\]|\d+", "", ev).strip(" ;")
    elif task == "truth":
        how = "stated the common misconception"
    elif task == "unanswerable":
        how = "invented details about something that does not exist"
    else:
        phrase = ev.split(":", 1)[1].strip(" '\"") if "refusal phrase" in ev else ""
        how = "refused a harmless request" + (f" ({phrase})" if phrase else "")
    extra = []
    if gen.truncated:
        extra.append("ran to the token limit")
    if _looping(gen.text):
        extra.append("repeats itself")
    return how + ("; " + ", ".join(extra) if extra else "")


def _looping(text: str) -> bool:
    from ..scorers.score import _repetitive

    return _repetitive(text)


def structural(gen: Generation) -> list[float]:
    t = gen.text
    words = re.findall(r"\w+", t)
    return [
        min(np.log1p(len(words)) / 6.0, 1.0),
        float(gen.truncated),
        float(t.rstrip()[-1:] in ".!?\"'}]" if t.strip() else 0),
        sum(ch.isdigit() for ch in t) / max(len(t), 1),
    ]


def _unit(a: np.ndarray) -> np.ndarray:
    return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)


def _best_kmeans(X: np.ndarray, k_range: tuple[int, int], seed: int) -> tuple[int, float, np.ndarray]:
    best: tuple[int, float, np.ndarray] | None = None
    for k in range(k_range[0], min(k_range[1], len(X) - 1) + 1):
        lab = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)
        if len(set(lab)) < 2:
            continue
        s = float(silhouette_score(X, lab))
        if best is None or s > best[1]:
            best = (k, s, lab)
    if best is None:
        return 1, 0.0, np.zeros(len(X), dtype=int)
    return best


def _purity(modes: list[str], labels: np.ndarray) -> float:
    total = 0
    for c in set(labels.tolist()):
        total += max(Counter(m for m, lab in zip(modes, labels, strict=True) if lab == c).values())
    return total / len(modes)


def _evaluate(X: np.ndarray, modes: list[str], tasks: list[str], k_range: tuple[int, int], seed: int) -> dict[str, Any]:
    k, sil, lab = _best_kmeans(X, k_range, seed)
    lab_mode = {c: Counter(m for m, x in zip(modes, lab, strict=True) if x == c).most_common(1)[0][0] for c in set(lab.tolist())}
    return {
        "k": int(k), "silhouette": round(sil, 3), "purity": round(_purity(modes, lab), 3),
        "ami_vs_mode": round(float(adjusted_mutual_info_score(modes, lab)), 3),
        "ami_vs_task": round(float(adjusted_mutual_info_score(tasks, lab)), 3),
        "mislabelled": sum(modes[n] != lab_mode[lab[n]] for n in range(len(modes))),
        "_labels": lab, "_lab_mode": lab_mode,
    }


# ---------------------------------------------------------------- layout helpers

def _pack(radii: list[float], aspect: float = 1.0) -> list[tuple[float, float]]:
    """Greedy circle packing around the origin. The cost of a spot is its distance after squeezing x by
    1/aspect, so a large aspect gives a wide layout. Circles never overlap."""
    order = sorted(range(len(radii)), key=lambda i: -radii[i])
    pos: dict[int, tuple[float, float]] = {}
    for i in order:
        r = radii[i]
        if not pos:
            pos[i] = (0.0, 0.0)
            continue
        best, best_cost = (0.0, 0.0), math.inf
        for deg in range(0, 360, 3):
            a = math.radians(deg)
            dx, dy = math.cos(a) * aspect, math.sin(a)
            t = 0.0
            while True:
                x, y = dx * t, dy * t
                if all(math.hypot(x - px, y - py) >= r + radii[j] for j, (px, py) in pos.items()):
                    break
                t += 0.25
            cost = math.hypot(x / aspect, y)
            if cost < best_cost:
                best, best_cost = (x, y), cost
        pos[i] = best
    return [pos[i] for i in range(len(radii))]


def _enclose(circles: list[tuple[float, float, float]]) -> tuple[float, float, float]:
    """Centre and radius of a small circle that contains all the given circles (x, y, r), found by nudging a
    centre toward the circle that sticks out furthest. Used so an island hugs the sub-clusters it contains."""
    cx = sum(x for x, _, _ in circles) / len(circles)
    cy = sum(y for _, y, _ in circles) / len(circles)
    for k in range(400):
        far = max(circles, key=lambda c: math.hypot(c[0] - cx, c[1] - cy) + c[2])
        w = 1.0 / (k + 2)
        cx, cy = cx + (far[0] - cx) * w * 0.5, cy + (far[1] - cy) * w * 0.5
    return cx, cy, max(math.hypot(x - cx, y - cy) + r for x, y, r in circles)


def _relax(P: np.ndarray, r: float, dmin: float = 1.15, iters: int = 200) -> np.ndarray:
    """Push points apart until no two are closer than dmin, keeping them inside a disc of radius r."""
    P = P.copy()
    for _ in range(iters):
        d = P[:, None, :] - P[None, :, :]
        dist = np.linalg.norm(d, axis=-1) + np.eye(len(P)) * 1e9
        close = dist < dmin
        if not close.any():
            break
        push = (d / dist[..., None]) * ((dmin - dist) / 2)[..., None] * close[..., None]
        P += push.sum(1)
        n = np.linalg.norm(P, axis=1)
        over = n > r
        P[over] *= (r / n[over])[:, None]
    return P


def _local_xy(X: np.ndarray, r: float, seed: int) -> np.ndarray:
    """2D positions inside a disc of radius r from the samples' own embedding (PCA), then relaxed."""
    n = len(X)
    if n == 1:
        return np.zeros((1, 2))
    Z = PCA(n_components=2, random_state=seed).fit_transform(X) if n > 2 else np.array([[-1.0, 0], [1.0, 0]])
    Z = Z - Z.mean(0)
    Z = Z / (np.percentile(np.linalg.norm(Z, axis=1), 90) + 1e-9) * (r * 0.75)
    Z += np.random.default_rng(seed).normal(0, 0.05, Z.shape)
    return _relax(Z, r - 0.6)


def _terms(docs: list[str], top: int = 2) -> list[list[str]]:
    """Distinctive words per document (each sub-cluster of an island is one document)."""
    try:
        tv = TfidfVectorizer(stop_words="english", max_features=800, token_pattern=r"(?u)\b[a-zA-Z]{3,}\b")
        M = tv.fit_transform(docs)
    except ValueError:
        return [[] for _ in docs]
    names = tv.get_feature_names_out()
    out = []
    for i in range(len(docs)):
        row = M[i].toarray().ravel()
        out.append([str(names[j]) for j in np.argsort(-row)[:top] if row[j] > 0 and names[j] not in ENGLISH_STOP_WORDS])
    return out


def cluster_failures(
    items: list[Item], gens: list[Generation], verdicts: list[Verdict], embed: EmbedFn = minilm_embed,
    seed: int = 7, k_range: tuple[int, int] = (3, 12),
) -> dict[str, Any]:
    gi = {(g.ckpt, g.item_id): g for g in gens}
    ii = {i.id: i for i in items}
    fails = [v for v in verdicts if not v.passed]
    if len(fails) < 4:
        return {"clusters": [], "islands": [], "points": [], "stats": {"n_failures": len(fails)}}
    modes = [v.mode or "wrong_answer" for v in fails]
    tasks = [v.task for v in fails]
    fgen = [gi[(v.ckpt, v.item_id)] for v in fails]
    feat = 0.35 * np.array([structural(g) for g in fgen])

    # representation 1: plain "[task] output"
    plain = np.hstack([embed([failure_text(ii[v.item_id], g) for v, g in zip(fails, fgen, strict=True)]), feat])
    # representation 2: output embedding minus the mean embedding of all outputs for the same task
    all_gens = list(gens)
    E = embed([(g.text.strip()[:400] or "(empty)") for g in all_gens])
    by_task: dict[str, list[int]] = {}
    for n, g in enumerate(all_gens):
        by_task.setdefault(ii[g.item_id].task, []).append(n)
    mean = {t: E[idx].mean(0) for t, idx in by_task.items()}
    pos = {(g.ckpt, g.item_id): n for n, g in enumerate(all_gens)}
    centred = np.hstack([_unit(np.array([E[pos[(v.ckpt, v.item_id)]] - mean[v.task] for v in fails])), feat])

    desc = embed([describe_failure(ii[v.item_id], g, v) for v, g in zip(fails, fgen, strict=True)])
    reps = {"plain": plain, "task_centred": centred, "failure_text": np.hstack([desc, 0.7 * centred[:, :-4]])}
    ev = {name: _evaluate(X, modes, tasks, k_range, seed) for name, X in reps.items()}
    chosen = max(ev, key=lambda k: (ev[k]["ami_vs_mode"], ev[k]["purity"]))
    X, lab, lab_mode = reps[chosen], ev[chosen]["_labels"], ev[chosen]["_lab_mode"]
    mis = [modes[n] != lab_mode[lab[n]] for n in range(len(fails))]

    # raw, unsupervised projection of the chosen representation
    Z = PCA(n_components=min(30, X.shape[0] - 1, X.shape[1]), random_state=seed).fit_transform(X)
    if len(fails) >= 12:
        raw = TSNE(n_components=2, perplexity=min(30, max(5, len(fails) // 4)), init="pca", random_state=seed).fit_transform(Z)
    else:
        raw = Z[:, :2]
    raw = raw - raw.mean(0)
    raw = raw / (np.abs(raw).max(0) + 1e-9)

    # islands: one per detector mode, sub-clusters inside by embedding
    islands_raw: list[dict[str, Any]] = []
    sub_of = np.zeros(len(fails), dtype=int)
    sub_info: list[dict[str, Any]] = []
    for mode in [m for m, _ in Counter(modes).most_common()]:
        mem = [n for n in range(len(fails)) if modes[n] == mode]
        ks = (2, min(5, len(mem) // 7))
        lab_sub = _best_kmeans(X[mem], ks, seed)[2] if len(mem) >= 14 and ks[1] >= ks[0] else np.zeros(len(mem), dtype=int)
        groups = [[mem[i] for i in range(len(mem)) if lab_sub[i] == c] for c in sorted(set(lab_sub.tolist()))]
        terms = _terms([" ".join(fgen[n].text for n in g) or "empty" for g in groups])
        subs = []
        for g, tm in zip(groups, terms, strict=True):
            r = 0.85 * math.sqrt(len(g)) + 1.4
            top_task, top_n = Counter(tasks[n] for n in g).most_common(1)[0]
            subs.append({"members": g, "r": r, "terms": tm, "task": top_task, "task_share": round(top_n / len(g), 2),
                         "local": _local_xy(X[g], r, seed)})
        centres = _pack([s["r"] + 0.6 for s in subs], 1.0)
        ex, ey, enc = _enclose([(cx, cy, s["r"]) for (cx, cy), s in zip(centres, subs, strict=True)])
        centres = [(cx - ex, cy - ey) for cx, cy in centres]
        enc += 0.8
        first = len(sub_info)
        for (cx, cy), s in zip(centres, subs, strict=True):
            sid = len(sub_info)
            rel = {}
            for n, (lx, ly) in zip(s["members"], s["local"], strict=True):
                sub_of[n] = sid
                rel[n] = (cx + float(lx), cy + float(ly))
            name = f"{s['task']} · {', '.join(s['terms'])}" if s["terms"] else f"{s['task']} · {mode.replace('_', ' ')}"
            sub_info.append({"id": sid, "label": mode, "size": len(s["members"]), "task": s["task"], "task_share": s["task_share"],
                             "terms": s["terms"], "rel": rel, "rel_c": (cx, cy), "r": s["r"], "name": name})
        islands_raw.append({"mode": mode, "n": len(mem), "R": enc, "subs": list(range(first, len(sub_info)))})

    layouts: dict[str, Any] = {}
    for key, asp in (("sq", 1.0), ("wide", WIDE_ASPECT)):
        centres = _pack([i["R"] + 2.5 for i in islands_raw], asp)
        circles = [(cx, cy, i["R"]) for (cx, cy), i in zip(centres, islands_raw, strict=True)]
        x0, x1 = min(x - r for x, _, r in circles), max(x + r for x, _, r in circles)
        y0, y1 = min(y - r for _, y, r in circles), max(y + r for _, y, r in circles)
        s = 2 / max(x1 - x0, y1 - y0)
        layouts[key] = {"centres": centres, "o": ((x0 + x1) / 2, (y0 + y1) / 2), "s": s,
                        "half": [round((x1 - x0) * s / 2, 4), round((y1 - y0) * s / 2, 4)]}

    def place(key: str, isl_idx: int, rel: tuple[float, float]) -> tuple[float, float]:
        L = layouts[key]
        ix, iy = L["centres"][isl_idx]
        return round((ix + rel[0] - L["o"][0]) * L["s"], 4), round((iy + rel[1] - L["o"][1]) * L["s"], 4)

    isl_of_sub = {sid: k for k, i in enumerate(islands_raw) for sid in i["subs"]}
    points = []
    for n, v in enumerate(fails):
        sid = int(sub_of[n])
        k = isl_of_sub[sid]
        x, y = place("sq", k, sub_info[sid]["rel"][n])
        wx, wy = place("wide", k, sub_info[sid]["rel"][n])
        points.append({"ckpt": v.ckpt, "item": v.item_id, "task": v.task, "mode": modes[n], "cluster": sid, "g": int(lab[n]),
                       "x": x, "y": y, "wx": wx, "wy": wy, "rx": round(float(raw[n, 0]), 4), "ry": round(float(raw[n, 1]), 4),
                       "mislabelled": bool(mis[n])})

    clusters = []
    for info in sub_info:
        k = isl_of_sub[info["id"]]
        mem = [n for n in range(len(fails)) if sub_of[n] == info["id"]]
        row: dict[str, Any] = {
            "id": info["id"], "label": info["label"], "name": info["name"], "size": info["size"], "task": info["task"],
            "task_share": info["task_share"], "terms": info["terms"],
            "by_ckpt": dict(Counter(fails[n].ckpt for n in mem)), "by_task": dict(Counter(tasks[n] for n in mem)),
            "disagree": int(sum(mis[n] for n in mem)),
        }
        for key, pre in (("sq", ""), ("wide", "w")):
            cx, cy = place(key, k, info["rel_c"])
            row[pre + "x"], row[pre + "y"], row[pre + "r"] = cx, cy, round(info["r"] * layouts[key]["s"], 4)
        clusters.append(row)
    islands = []
    for k, i in enumerate(islands_raw):
        row = {"mode": i["mode"], "n": i["n"], "disagree": int(sum(mis[n] for n in range(len(fails)) if modes[n] == i["mode"])),
               "subs": i["subs"]}
        for key, pre in (("sq", ""), ("wide", "w")):
            cx, cy = place(key, k, (0.0, 0.0))
            row[pre + "x"], row[pre + "y"], row[pre + "r"] = cx, cy, round(i["R"] * layouts[key]["s"], 4)
        islands.append(row)

    globals_ = []
    for c in sorted(lab_mode):
        votes = Counter(m for m, x in zip(modes, lab, strict=True) if x == c)
        globals_.append({"id": int(c), "label": lab_mode[c], "size": int(sum(votes.values())),
                         "purity": round(votes.most_common(1)[0][1] / sum(votes.values()), 3), "votes": dict(votes)})
    pub = {name: {k: v for k, v in e.items() if not k.startswith("_")} for name, e in ev.items()}
    c = pub[chosen]
    stats = {
        "n_failures": len(fails), "chosen": chosen, "chosen_by": "higher adjusted mutual information with the detector mode",
        "k": c["k"], "silhouette": c["silhouette"], "purity": c["purity"], "ami_vs_mode": c["ami_vs_mode"],
        "ami_vs_task": c["ami_vs_task"], "mislabelled": c["mislabelled"], "representations": pub,
        "ami_mode_vs_task": round(float(adjusted_mutual_info_score(modes, tasks)), 3),
        "extent": {k: L["half"] for k, L in layouts.items()}, "n_islands": len(islands), "n_sub": len(clusters),
        "embedder": "all-MiniLM-L6-v2; three representations compared (plain, task-centred, failure description); KMeans (k by silhouette) "
                    "for purity; islands by detector mode, sub-clusters by KMeans inside each island; raw t-SNE of the chosen representation",
    }
    return {"clusters": clusters, "islands": islands, "points": points, "global": globals_, "stats": stats}
