"""Redis-backed job queue shared by the API (producer) and parse_worker.py (consumer).

The worker also publishes a short-lived heartbeat (`<queue>:worker`) so the API can
tell the UI whether the worker is loading models, idle, parsing, or not running.
"""
import json
import time
from typing import Any

import redis as syncredis
import redis.asyncio as aioredis

from .config import settings

_WORKER_KEY = f"{settings.parse_queue}:worker"
_HEARTBEAT_TTL_S = 30


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


def set_worker_status(status: dict[str, Any]) -> None:
    """Worker side: publish its state. Expires automatically if the worker dies."""
    try:
        r = syncredis.from_url(settings.redis_url, socket_connect_timeout=2, socket_timeout=3)
        try:
            r.set(_WORKER_KEY, json.dumps({**status, "at": time.time()}), ex=_HEARTBEAT_TTL_S)
        finally:
            r.close()
    except Exception:  # noqa: BLE001 - a missing heartbeat must never break parsing
        pass


async def get_worker_status() -> dict[str, Any]:
    """API side: the worker's last heartbeat, or state=offline when there is none."""
    r = aioredis.from_url(settings.redis_url, socket_connect_timeout=2, socket_timeout=3)
    try:
        raw = await r.get(_WORKER_KEY)
        pending = await r.llen(settings.parse_queue)
    except Exception:  # noqa: BLE001
        return {"state": "unreachable", "queued_jobs": 0}
    finally:
        await r.aclose()
    status: dict[str, Any] = json.loads(raw) if raw else {"state": "offline"}
    status["queued_jobs"] = int(pending or 0)
    return status
