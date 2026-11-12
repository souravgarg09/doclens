"""Okapi BM25 over a document's regions, used for hybrid retrieval.

SigLIP's text tower only reads the first 64 tokens of a region, so long paragraphs
and dense tables are poorly represented by the vector alone. BM25 matches exact
terms and numbers anywhere in a region ("hypertrophy", "0.24", "Table 2"). The two
rankings are fused with reciprocal-rank fusion in retriever.py.

Pure Python: a document has a few hundred regions, so scoring all of them per
question takes a few milliseconds and needs no extra index or service.
"""
from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
_STOPWORDS = frozenset(
    "a an and are as at be by for from has have how in into is it its of on or that the their this to was "
    "were what when where which who why will with does did do than then there these those".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall((text or "").lower()) if t not in _STOPWORDS]


def bm25_scores(query: str, documents: list[str], k1: float = 1.5, b: float = 0.75) -> list[float]:
    """BM25 score of every document for the query (same order as `documents`)."""
    q_terms = set(tokenize(query))
    if not q_terms or not documents:
        return [0.0] * len(documents)
    tokenized = [tokenize(d) for d in documents]
    n = len(tokenized)
    avg_len = sum(len(t) for t in tokenized) / n or 1.0
    df: Counter[str] = Counter()
    for toks in tokenized:
        df.update(set(toks) & q_terms)
    idf = {t: math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5)) for t in q_terms}
    scores = []
    for toks in tokenized:
        tf = Counter(toks)
        length_norm = k1 * (1 - b + b * len(toks) / avg_len)
        scores.append(sum(idf[t] * tf[t] * (k1 + 1) / (tf[t] + length_norm) for t in q_terms if tf[t]))
    return scores


def reciprocal_rank_fusion(rankings: list[list[int]], weights: list[float] | None = None, k: int = 60) -> dict[int, float]:
    """Fuse ranked id lists: score(id) = Σ weight / (k + rank). Higher is better."""
    weights = weights or [1.0] * len(rankings)
    fused: dict[int, float] = {}
    for ranking, weight in zip(rankings, weights):
        for rank, item in enumerate(ranking, start=1):
            fused[item] = fused.get(item, 0.0) + weight / (k + rank)
    return fused
