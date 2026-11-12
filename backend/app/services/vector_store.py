"""Per-document region index stored in Qdrant.

Vectors from different embedding models are not comparable, so each embedding model
gets its own collection: "<QDRANT_COLLECTION>__<model slug>". Changing SIGLIP_MODEL
therefore never mixes old and new vectors; the worker re-parses documents that have
no vectors for the current model (see parse_worker.reindex_stale_documents) and
removes collections left over from previous models.
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


def collection_name() -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", settings.siglip_model.lower()).strip("_")
    return f"{settings.qdrant_collection}__{slug}"


def ensure_collection(dim: int) -> None:
    c = client()
    name = collection_name()
    if not c.collection_exists(name):
        c.create_collection(name, vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE))
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


def upsert_regions(points: list[tuple[int, list[float], dict[str, Any]]]) -> None:
    client().upsert(collection_name(),
                    points=[qm.PointStruct(id=pid, vector=vec, payload=payload) for pid, vec, payload in points])


def delete_document(document_id: int) -> None:
    c = client()
    name = collection_name()
    if not c.collection_exists(name):
        return
    c.delete(name, points_selector=qm.FilterSelector(filter=qm.Filter(
        must=[qm.FieldCondition(key="document_id", match=qm.MatchValue(value=document_id))])))


def search(document_id: int, vector: list[float], limit: int, types: list[str] | None = None) -> list[dict[str, Any]]:
    c = client()
    name = collection_name()
    if not c.collection_exists(name):
        return []
    must: list[Any] = [qm.FieldCondition(key="document_id", match=qm.MatchValue(value=document_id))]
    if types:
        must.append(qm.FieldCondition(key="type", match=qm.MatchAny(any=types)))
    hits = c.query_points(name, query=vector, query_filter=qm.Filter(must=must), limit=limit, with_payload=True).points
    return [{"region_id": int(h.id), "score": float(h.score), **(h.payload or {})} for h in hits]
