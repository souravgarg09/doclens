"""Per-document region index stored in Qdrant.

Each region point has up to two named vectors: "visual" (SigLIP 2, image + text) and
"text" (the text embedder). Vectors from different embedding models are not comparable,
so each model combination gets its own collection: "<QDRANT_COLLECTION>__<models slug>".
Changing SIGLIP_MODEL or TEXT_EMBED_MODEL therefore never mixes old and new vectors; the
worker re-parses documents that have no vectors in the current collection (see
parse_worker.reindex_stale_documents) and removes collections left over from earlier models.
"""
import re
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from ..config import settings

_client: QdrantClient | None = None


def client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(url=settings.qdrant_url)
    return _client


VISUAL, TEXT = "visual", "text"


def _slug(model: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", model.lower()).strip("_")


def collection_name() -> str:
    name = f"{settings.qdrant_collection}__{_slug(settings.siglip_model)}"
    if settings.text_embed_model.strip():
        name += f"__{_slug(settings.text_embed_model.split('/')[-1])}"
    return name


def ensure_collection(visual_dim: int, text_dim: int | None = None) -> None:
    c = client()
    name = collection_name()
    if not c.collection_exists(name):
        vectors = {VISUAL: qm.VectorParams(size=visual_dim, distance=qm.Distance.COSINE)}
        if text_dim:
            vectors[TEXT] = qm.VectorParams(size=text_dim, distance=qm.Distance.COSINE)
        c.create_collection(name, vectors_config=vectors)
        c.create_payload_index(name, "document_id", qm.PayloadSchemaType.INTEGER)


def stale_collections() -> list[str]:
    """Collections created for embedding models that are no longer configured."""
    current = collection_name()
    base = settings.qdrant_collection
    return [col.name for col in client().get_collections().collections
            if col.name != current and (col.name == base or col.name.startswith(f"{base}__"))]


def drop_collection(name: str) -> None:
    client().delete_collection(name)


def count_document(document_id: int) -> int:
    c = client()
    name = collection_name()
    if not c.collection_exists(name):
        return 0
    flt = qm.Filter(must=[qm.FieldCondition(key="document_id", match=qm.MatchValue(value=document_id))])
    return int(c.count(name, count_filter=flt, exact=True).count)


def upsert_regions(points: list[tuple[int, dict[str, list[float]], dict[str, Any]]]) -> None:
    """points: (region id, {"visual": [...], "text": [...]}, payload)."""
    client().upsert(collection_name(),
                    points=[qm.PointStruct(id=pid, vector=vec, payload=payload) for pid, vec, payload in points])


def delete_document(document_id: int) -> None:
    c = client()
    name = collection_name()
    if not c.collection_exists(name):
        return
    c.delete(name, points_selector=qm.FilterSelector(filter=qm.Filter(
        must=[qm.FieldCondition(key="document_id", match=qm.MatchValue(value=document_id))])))


def search(document_id: int, vector: list[float], limit: int, types: list[str] | None = None,
           using: str = VISUAL) -> list[dict[str, Any]]:
    c = client()
    name = collection_name()
    if not c.collection_exists(name):
        return []
    must: list[Any] = [qm.FieldCondition(key="document_id", match=qm.MatchValue(value=document_id))]
    if types:
        must.append(qm.FieldCondition(key="type", match=qm.MatchAny(any=types)))
    hits = c.query_points(name, query=vector, using=using, query_filter=qm.Filter(must=must), limit=limit,
                          with_payload=True).points
    return [{"region_id": int(h.id), "score": float(h.score), **(h.payload or {})} for h in hits]
