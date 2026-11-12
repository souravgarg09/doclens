"""Router + retrieval: embed the question with SigLIP text tower and search the
per-document region index; hydrate hits with region rows from Postgres."""
import asyncio
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import settings
from ..db import Region
from . import embeddings, vector_store


async def retrieve(session: AsyncSession, document_id: int, question: str) -> list[dict[str, Any]]:
    qvec = (await asyncio.to_thread(embeddings.embed_text, [question]))[0].tolist()
    hits = await asyncio.to_thread(vector_store.search, document_id, qvec, settings.top_k_retrieve)
    if not hits:
        return []
    ids = [h["region_id"] for h in hits]
    rows = (await session.execute(select(Region).where(Region.id.in_(ids)))).scalars().all()
    by_id = {r.id: r.as_dict(include_paths=True) for r in rows}
    out = []
    for h in hits:
        r = by_id.get(h["region_id"])
        if r:
            out.append({**r, "region_id": r["id"], "score": h["score"]})
    return out
