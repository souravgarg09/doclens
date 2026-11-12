"""Redis-backed job queue shared by the API (producer) and parse_worker.py (consumer)."""
import json
import redis.asyncio as aioredis
import redis as syncredis

from .config import settings


async def enqueue_parse(document_id: int) -> None:
    r = aioredis.from_url(settings.redis_url, socket_connect_timeout=2, socket_timeout=3)
    try:
        await r.lpush(settings.parse_queue, json.dumps({"document_id": document_id}))
    finally:
        await r.aclose()


def blocking_pop(timeout: int = 5) -> dict | None:
    r = syncredis.from_url(settings.redis_url, socket_connect_timeout=2, socket_timeout=timeout + 2)
    try:
        item = r.brpop(settings.parse_queue, timeout=timeout)
        return json.loads(item[1]) if item else None
    finally:
        r.close()
