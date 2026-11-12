"""Hybrid retrieval: SigLIP vectors (Qdrant) fused with BM25 keyword scores.

1. Embed the question with the SigLIP text tower and score the document's regions
   by cosine similarity in Qdrant (catches visual/semantic matches, e.g. a chart).
2. Score the same regions with BM25 (catches exact terms and numbers anywhere in a
   long paragraph or table, which SigLIP's 64-token text window misses).
3. Fuse both rankings with reciprocal-rank fusion and keep the top `top_k_retrieve`.

Each candidate keeps its SigLIP cosine as `score` (the reranker and grounding
guardrail are calibrated on it) plus `lexical` and `fused` for the trace.
"""
import asyncio
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import settings
from ..db import Region
from . import embeddings, lexical, vector_store

# Cosine scores are fetched for (almost) every region so that a region found only
# by BM25 still carries a calibrated similarity score.
_DENSE_POOL = 512


async def retrieve(session: AsyncSession, document_id: int, question: str) -> list[dict[str, Any]]:
    qvec = (await asyncio.to_thread(embeddings.embed_text, [question]))[0].tolist()
    hits = await asyncio.to_thread(vector_store.search, document_id, qvec, _DENSE_POOL)
    if not hits:
        return []
    cosine = {h["region_id"]: h["score"] for h in hits}
    dense_ranking = [h["region_id"] for h in hits]  # already sorted by similarity

    rows = (await session.execute(select(Region).where(Region.document_id == document_id))).scalars().all()
    by_id = {r.id: r for r in rows}
    ids = [r.id for r in rows]
    bm25 = lexical.bm25_scores(question, [f"{r.type} {r.content or ''}" for r in rows])
    lexical_score = dict(zip(ids, bm25))
    lexical_ranking = [i for i, s in sorted(lexical_score.items(), key=lambda kv: kv[1], reverse=True) if s > 0]

    fused = lexical.reciprocal_rank_fusion(
        [dense_ranking[: settings.top_k_retrieve * 2], lexical_ranking[: settings.top_k_retrieve * 2]],
        weights=[1.0, settings.lexical_weight],
    )
    ranked = sorted(fused, key=fused.get, reverse=True)[: settings.top_k_retrieve]

    out = []
    for region_id in ranked:
        row = by_id.get(region_id)
        if row is None:
            continue
        r = row.as_dict(include_paths=True)
        out.append({**r, "region_id": r["id"], "score": float(cosine.get(region_id, 0.0)),
                    "lexical": round(float(lexical_score.get(region_id, 0.0)), 3),
                    "fused": round(fused[region_id], 5)})
    return out
