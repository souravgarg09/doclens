"""Gemini Interactions API wrapper with text, image, and JSON support.

Adds exponential-backoff retry for transient server failures and switches to a
higher-throughput fallback model when the preferred model is rate limited.
"""
import base64
import json
import logging
import random
import re
import time
from pathlib import Path
from typing import Any, Callable, TypeVar

from google import genai
from ..config import settings

log = logging.getLogger("gemini")

_client: genai.Client | None = None

# HTTP statuses worth retrying. Anything else (400 bad request, 401/403 auth,
# 404 unknown model) is a real bug and should surface immediately.
_RETRY_STATUS = {500, 502, 503, 504}
_MAX_ATTEMPTS = 5
_BASE_DELAY_S = 1.5

T = TypeVar("T")


class GeminiUnavailable(RuntimeError):
    """Raised when Gemini stays unavailable after all retries.

    Callers (graph.py) can catch this to report "model unavailable, try again"
    instead of misreporting it as "not enough evidence".
    """


def client() -> genai.Client:
    global _client
    if _client is None:
        if not settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        _client = genai.Client(api_key=settings.gemini_api_key)
    return _client


def image_part(path: str | Path) -> dict[str, str]:
    p = settings.resolve_project_path(path)
    return {
        "type": "image",
        "mime_type": "image/png",
        "data": base64.b64encode(p.read_bytes()).decode("ascii"),
    }


def _status_code(e: Exception) -> int | None:
    for attr in ("code", "status_code"):
        v = getattr(e, attr, None)
        if isinstance(v, int):
            return v
    return None


def _with_retry(fn: Callable[[], T], *, what: str = "gemini", model: str = "") -> T:
    """Run fn(); on transient APIError retry with exponential backoff + jitter."""
    last: Exception | None = None
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            return fn()
        except Exception as e:  # SDK exposes different error classes per API surface.
            code = _status_code(e)
            if code == 429:
                last = e
                log.warning("%s: model %s is rate limited; trying configured fallback", what, model)
                break
            if code not in _RETRY_STATUS:
                raise  # permanent error — don't hide it
            last = e
            if attempt == _MAX_ATTEMPTS:
                break
            wait = _BASE_DELAY_S * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
            log.warning("%s: HTTP %s on attempt %d/%d — retrying in %.1fs",
                        what, code, attempt, _MAX_ATTEMPTS, wait)
            time.sleep(wait)
    raise GeminiUnavailable(
        f"{what}: model '{model or settings.gemini_model}' unavailable ({last})"
    ) from last


def generate(parts: list[Any], *, system: str | None = None,
             json_mode: bool = True, schema: dict[str, Any] | None = None) -> str:
    inputs = [
        {"type": "text", "text": part} if isinstance(part, str) else part
        for part in parts
    ]
    response_format = None
    if json_mode:
        response_format = {
            "type": "text",
            "mime_type": "application/json",
        }
        if schema:
            response_format["schema"] = schema
    models = [settings.gemini_model]
    fallback = settings.gemini_fallback_model.strip()
    if fallback and fallback not in models:
        models.append(fallback)

    failures: list[str] = []
    for model in models:
        try:
            resp = _with_retry(
                lambda: client().interactions.create(
                    model=model,
                    input=inputs,
                    system_instruction=system,
                    response_format=response_format,
                    store=False,
                    timeout=60,
                ),
                what="generate",
                model=model,
            )
            if model != settings.gemini_model:
                log.info("generate: used fallback model %s", model)
            return resp.output_text or ""
        except GeminiUnavailable as exc:
            failures.append(str(exc))
    raise GeminiUnavailable("; ".join(failures))


def parse_json(raw: str) -> Any:
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        s = min([i for i in (raw.find("{"), raw.find("[")) if i >= 0] or [0])
        e = max(raw.rfind("}"), raw.rfind("]"))
        return json.loads(raw[s:e + 1])
