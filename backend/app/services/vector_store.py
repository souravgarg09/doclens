"""Per-document region index stored in Qdrant."""
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


def ensure_collection(dim: int) -> None:
    c = client()
    if not c.collection_exists(settings.qdrant_collection):
        c.create_collection(settings.qdrant_collection, vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE))
        c.create_payload_index(settings.qdrant_collection, "document_id", qm.PayloadSchemaType.INTEGER)


def upsert_regions(points: list[tuple[int, list[float], dict[str, Any]]]) -> None:
    client().upsert(settings.qdrant_collection, points=[qm.PointStruct(id=pid, vector=vec, payload=payload) for pid, vec, payload in points])


def delete_document(document_id: int) -> None:
    c = client()
    if not c.collection_exists(settings.qdrant_collection):
        return
    c.delete(settings.qdrant_collection, points_selector=qm.FilterSelector(filter=qm.Filter(must=[qm.FieldCondition(key="document_id", match=qm.MatchValue(value=document_id))])))


def search(document_id: int, vector: list[float], limit: int, types: list[str] | None = None) -> list[dict[str, Any]]:
    c = client()
    if not c.collection_exists(settings.qdrant_collection):
        return []
    must: list[Any] = [qm.FieldCondition(key="document_id", match=qm.MatchValue(value=document_id))]
    if types:
        must.append(qm.FieldCondition(key="type", match=qm.MatchAny(any=types)))
    hits = c.query_points(settings.qdrant_collection, query=vector, query_filter=qm.Filter(must=must), limit=limit, with_payload=True).points
    return [{"region_id": int(h.id), "score": float(h.score), **(h.payload or {})} for h in hits]
