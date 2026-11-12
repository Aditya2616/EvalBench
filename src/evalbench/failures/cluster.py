"""Embed failing samples, cluster them, lay them out in 2D and label the clusters.

Labels come from the deterministic failure-mode tags (plurality vote), not from a model. Cluster
purity is reported, and every sample whose own tag disagrees with its cluster's label is listed.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_mutual_info_score, silhouette_score

from ..schema import Generation, Item, Verdict

EmbedFn = Callable[[list[str]], np.ndarray]
_MODEL = None


def minilm_embed(texts: list[str]) -> np.ndarray:
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer

        _MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    return np.asarray(_MODEL.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False))


def failure_text(item: Item, gen: Generation) -> str:
    return f"[{item.task}] {gen.text.strip()[:400]}"


def structural(gen: Generation) -> list[float]:
    t = gen.text
    words = re.findall(r"\w+", t)
    return [
        min(np.log1p(len(words)) / 6.0, 1.0),
        float(gen.truncated),
        float(t.rstrip()[-1:] in ".!?\"'}]" if t.strip() else 0),
        sum(ch.isdigit() for ch in t) / max(len(t), 1),
    ]


def cluster_failures(
    items: list[Item], gens: list[Generation], verdicts: list[Verdict], embed: EmbedFn = minilm_embed,
    seed: int = 7, k_range: tuple[int, int] = (3, 8),
) -> dict[str, Any]:
    gi = {(g.ckpt, g.item_id): g for g in gens}
    ii = {i.id: i for i in items}
    fails = [v for v in verdicts if not v.passed]
    if len(fails) < 4:
        return {"clusters": [], "points": [], "stats": {"n_failures": len(fails)}}
    emb = embed([failure_text(ii[v.item_id], gi[(v.ckpt, v.item_id)]) for v in fails])
    feat = np.array([structural(gi[(v.ckpt, v.item_id)]) for v in fails])
    X = np.hstack([emb, 0.35 * feat])

    best_k, best_s, best_labels = None, -2.0, None
    for k in range(k_range[0], min(k_range[1], len(fails) - 1) + 1):
        lab = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)
        s = silhouette_score(X, lab)
        if s > best_s:
            best_k, best_s, best_labels = k, s, lab
    labels = np.asarray(best_labels)

    n_comp = min(30, X.shape[0] - 1, X.shape[1])
    Z = PCA(n_components=n_comp, random_state=seed).fit_transform(X)
    if len(fails) >= 12:
        xy = TSNE(n_components=2, perplexity=min(30, max(5, len(fails) // 4)), init="pca", random_state=seed).fit_transform(Z)
    else:
        xy = Z[:, :2]
    xy = (xy - xy.mean(0)) / (np.abs(xy - xy.mean(0)).max() + 1e-9)

    modes = [v.mode or "wrong_answer" for v in fails]
    clusters: list[dict[str, Any]] = []
    used: Counter[str] = Counter()
    for c in range(best_k):
        mem = [n for n in range(len(fails)) if labels[n] == c]
        votes = Counter(modes[n] for n in mem)
        label, cnt = votes.most_common(1)[0]
        used[label] += 1
        texts = [gi[(fails[n].ckpt, fails[n].item_id)].text or "(empty)" for n in mem]
        try:
            tv = TfidfVectorizer(stop_words="english", max_features=300, token_pattern=r"(?u)\b\w+\b")
            m = tv.fit_transform(texts).sum(0).A1
            terms = [tv.get_feature_names_out()[i] for i in np.argsort(-m)[:5] if m[i] > 0]
        except ValueError:
            terms = []
        clusters.append({
            "id": int(c), "label": label, "sub": used[label], "size": len(mem), "purity": round(cnt / len(mem), 3),
            "votes": dict(votes), "terms": terms,
            "by_ckpt": dict(Counter(fails[n].ckpt for n in mem)),
            "by_task": dict(Counter(fails[n].task for n in mem)),
            "centroid": [float(xy[mem, 0].mean()), float(xy[mem, 1].mean())],
        })
    totals = Counter(cl["label"] for cl in clusters)
    for cl in clusters:  # disambiguate when two clusters share a mode
        cl["name"] = cl["label"].upper().replace("_", " ") + (f" {chr(64 + cl['sub'])}" if totals[cl["label"]] > 1 else "")
    lab_of = {cl["id"]: cl["label"] for cl in clusters}
    points = []
    for n, v in enumerate(fails):
        c = int(labels[n])
        points.append({"ckpt": v.ckpt, "item": v.item_id, "task": v.task, "mode": modes[n], "cluster": c,
                       "x": round(float(xy[n, 0]), 4), "y": round(float(xy[n, 1]), 4), "mislabelled": modes[n] != lab_of[c]})
    tasks = [v.task for v in fails]
    pur = sum(max(Counter(modes[n] for n in range(len(fails)) if labels[n] == c).values()) for c in range(best_k)) / len(fails)
    stats = {
        "n_failures": len(fails), "k": int(best_k), "silhouette": round(float(best_s), 3), "purity": round(pur, 3),
        "ami_vs_mode": round(float(adjusted_mutual_info_score(modes, labels)), 3),
        "ami_vs_task": round(float(adjusted_mutual_info_score(tasks, labels)), 3),
        "ami_mode_vs_task": round(float(adjusted_mutual_info_score(modes, tasks)), 3),
        "mislabelled": sum(p["mislabelled"] for p in points),
        "embedder": "all-MiniLM-L6-v2 + 4 structural features, KMeans (k by silhouette), t-SNE layout",
    }
    return {"clusters": clusters, "points": points, "stats": stats}
