"""Hybrid retrieval: text vectors + BM25 keywords (+ a small SigLIP 2 vote for visual regions).

1. Text embedder (arctic-embed-s, 512 tokens): ranks regions by what they say, on meaning.
2. BM25: exact words and numbers anywhere in a region ("Table 2", "0.24", "Hochreiter"),
   including the targets of the region's hidden PDF links.
3. SigLIP 2 (image + 64-token text): ranks tables / charts / figures / equations by how they
   look, with a small weight (`visual_weight`), so a chart with little text can still be found.
4. The rankings are fused with reciprocal-rank fusion; the top `top_k_retrieve` are kept.

Text vectors + BM25 rank paragraphs far better than SigLIP alone; SigLIP keeps a small vote
because a larger one pulls text results down.

Each candidate keeps `score` = text-embedder cosine (the reranker's relevance and the grounding
guardrail are calibrated on it), plus `visual`, `lexical` and `fused` for the trace.
With TEXT_EMBED_MODEL empty, retrieval falls back to SigLIP 2 + BM25 and `score` is SigLIP's.
"""
import asyncio
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import settings
from ..db import Link, Region
from . import embeddings, lexical, vector_store

# Cosine scores are fetched for (almost) every region so that a region found only
# by BM25 still carries a calibrated similarity score.
_DENSE_POOL = 512
_VISUAL_TYPES = ["table", "chart", "figure", "equation"]


async def retrieve(session: AsyncSession, document_id: int, question: str) -> list[dict[str, Any]]:
    use_text = embeddings.text_enabled()
    rankings: list[list[int]] = []
    weights: list[float] = []
    text_cos: dict[int, float] = {}
    visual_cos: dict[int, float] = {}

    if use_text:
        qtext = await asyncio.to_thread(embeddings.embed_query, question)
        hits = await asyncio.to_thread(vector_store.search, document_id, qtext, _DENSE_POOL, None, vector_store.TEXT)
        text_cos = {h["region_id"]: h["score"] for h in hits}
        rankings.append([h["region_id"] for h in hits][: settings.top_k_retrieve * 2])
        weights.append(1.0)
    if not use_text or settings.visual_weight > 0:
        qvis = (await asyncio.to_thread(embeddings.embed_text, [question]))[0].tolist()
        hits = await asyncio.to_thread(vector_store.search, document_id, qvis, _DENSE_POOL,
                                       _VISUAL_TYPES if use_text else None, vector_store.VISUAL)
        visual_cos = {h["region_id"]: h["score"] for h in hits}
        rankings.append([h["region_id"] for h in hits][: settings.top_k_retrieve * 2])
        weights.append(settings.visual_weight if use_text else 1.0)
    if not text_cos and not visual_cos:
        return []

    rows = (await session.execute(select(Region).where(Region.document_id == document_id))).scalars().all()
    by_id = {r.id: r for r in rows}
    ids = [r.id for r in rows]
    # a region's hidden link targets count as its words too: "the code on GitHub" finds the
    # sentence whose word "code" links to github.com, "Hochreiter" finds the passage citing him
    link_words: dict[int, list[str]] = {}
    for region_id, uri, target_text, kind in (await session.execute(
            select(Link.region_id, Link.uri, Link.target_text, Link.kind)
            .where(Link.document_id == document_id, Link.kind != "reference"))).all():
        if region_id is not None and kind in ("citation", "url", "doi", "email"):
            link_words.setdefault(region_id, []).append(uri or (target_text or "")[:90])
    bm25 = lexical.bm25_scores(question, [f"{r.type} {r.content or ''} {' '.join(link_words.get(r.id, []))}" for r in rows])
    lexical_score = dict(zip(ids, bm25))
    rankings.append([i for i, s in sorted(lexical_score.items(), key=lambda kv: kv[1], reverse=True) if s > 0]
                    [: settings.top_k_retrieve * 2])
    weights.append(settings.lexical_weight)

    fused = lexical.reciprocal_rank_fusion(rankings, weights=weights)
    ranked = sorted(fused, key=lambda i: fused[i], reverse=True)[: settings.top_k_retrieve]
    primary = text_cos if use_text else visual_cos

    out = []
    for region_id in ranked:
        row = by_id.get(region_id)
        if row is None:
            continue
        r = row.as_dict(include_paths=True)
        out.append({**r, "region_id": r["id"], "score": float(primary.get(region_id, 0.0)),
                    "visual": round(float(visual_cos.get(region_id, 0.0)), 4),
                    "lexical": round(float(lexical_score.get(region_id, 0.0)), 3),
                    "fused": round(fused[region_id], 5)})
    return out
