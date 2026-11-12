"""Gemini Interactions API wrapper with text, image, and JSON support.

Failure policy (so a question can never hang for minutes):
  * every HTTP request has a hard client-side timeout (GEMINI_TIMEOUT_S);
  * a transient failure (5xx overload, timeout, dropped connection) is retried at most
    GEMINI_MAX_ATTEMPTS times per model with a short pause;
  * a rate limit (429) or a model that keeps failing moves straight to the fallback model;
  * the whole call stops starting new attempts once GEMINI_BUDGET_S has elapsed;
  * hedging: if a request has not answered after GEMINI_HEDGE_AFTER_S, an identical
    request is sent in parallel and whichever answers first wins. Gemini latency is
    bimodal (about 2 s or about 20 s for the same prompt), so this cuts the slow tail.
Callers get GeminiUnavailable, which the agent reports as "model unavailable, try again"
instead of a wrong "not in the document".
"""
import base64
import json
import logging
import random
import re
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Callable

from google import genai
from google.genai import types

from ..config import settings

log = logging.getLogger("gemini")

_client: genai.Client | None = None

# HTTP statuses worth retrying. Anything else (400 bad request, 401/403 auth,
# 404 unknown model) is a real bug and should surface immediately.
_RETRY_STATUS = {500, 502, 503, 504}
_TRANSIENT_ERROR_WORDS = ("timeout", "timed out", "connection", "remote protocol", "temporarily")
_POOL = ThreadPoolExecutor(max_workers=8, thread_name_prefix="gemini")


class GeminiUnavailable(RuntimeError):
    """Raised when every configured model failed within the time budget."""


def client() -> genai.Client:
    global _client
    if _client is None:
        if not settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        _client = genai.Client(
            api_key=settings.gemini_api_key,
            http_options=types.HttpOptions(timeout=int(settings.gemini_timeout_s * 1000)),  # milliseconds
        )
    return _client


def image_part(path: str | Path) -> dict[str, str]:
    p = settings.resolve_project_path(path)
    # Only region crops written by the parse worker may be sent to the model.
    if not p.is_relative_to(settings.crops_dir) or not p.is_file():
        raise OSError(f"refusing to send a file outside the crops directory: {p}")
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


def _classify(e: Exception) -> str:
    """'rate_limited', 'model_missing', 'transient' or 'permanent'."""
    code = _status_code(e)
    if code == 429:
        return "rate_limited"
    if code == 404:
        return "model_missing"  # retired or misspelled model name: try the next model
    if code in _RETRY_STATUS:
        return "transient"
    text = f"{type(e).__name__} {e}".lower()
    if code is None and any(word in text for word in _TRANSIENT_ERROR_WORDS):
        return "transient"
    return "permanent"


def _models() -> list[str]:
    models = [settings.gemini_model]
    fallback = settings.gemini_fallback_model.strip()
    if fallback and fallback not in models:
        models.append(fallback)
    return models


def _hedged(call: Callable[[], Any]) -> Any:
    """Run call(); if it is still running after the hedge delay, race a duplicate."""
    first = _POOL.submit(call)
    if settings.gemini_hedge_after_s <= 0:
        return first.result()
    done, _ = wait([first], timeout=settings.gemini_hedge_after_s)
    if done:
        return first.result()
    log.info("generate: no reply after %.0fs; sending a hedged duplicate request", settings.gemini_hedge_after_s)
    pending: set[Future] = {first, _POOL.submit(call)}
    error: BaseException | None = None
    while pending:
        finished, pending = wait(pending, return_when=FIRST_COMPLETED)
        for future in finished:
            if future.exception() is None:
                return future.result()  # the slower request finishes in the background and is ignored
            error = future.exception()
    assert error is not None
    raise error


def generate(parts: list[Any], *, system: str | None = None,
             json_mode: bool = True, schema: dict[str, Any] | None = None) -> str:
    inputs = [{"type": "text", "text": part} if isinstance(part, str) else part for part in parts]
    response_format: dict[str, Any] | None = None
    if json_mode:
        response_format = {"type": "text", "mime_type": "application/json"}
        if schema:
            response_format["schema"] = schema

    started = time.monotonic()
    failures: list[str] = []
    for model in _models():
        for attempt in range(1, settings.gemini_max_attempts + 1):
            if time.monotonic() - started > settings.gemini_budget_s:
                failures.append(f"{model}: time budget of {settings.gemini_budget_s:.0f}s used up")
                raise GeminiUnavailable("; ".join(failures))
            try:
                extra: dict[str, Any] = {}
                if settings.gemini_thinking:
                    extra["generation_config"] = {"thinking_level": settings.gemini_thinking}
                resp = _hedged(lambda: client().interactions.create(
                    model=model, input=inputs, system_instruction=system,
                    response_format=response_format, store=False, **extra,
                ))
                log.info("generate: %s answered in %.1fs%s", model, time.monotonic() - started,
                         "" if model == settings.gemini_model else " (fallback)")
                return resp.output_text or ""
            except Exception as e:  # noqa: BLE001 - the SDK raises different classes per API surface
                kind = _classify(e)
                if kind == "permanent":
                    raise
                failures.append(f"{model} attempt {attempt}: {type(e).__name__} {str(e)[:160]}")
                if kind == "rate_limited":
                    log.warning("generate: %s is rate limited (429); switching model", model)
                    break
                if kind == "model_missing":
                    log.error("generate: model %s was not found (404); check GEMINI_MODEL / "
                              "GEMINI_FALLBACK_MODEL. Trying the next model", model)
                    break
                log.warning("generate: %s transient failure on attempt %d/%d: %s",
                            model, attempt, settings.gemini_max_attempts, str(e)[:160])
                if attempt < settings.gemini_max_attempts:
                    time.sleep(1.0 + random.uniform(0, 0.5))
    raise GeminiUnavailable("; ".join(failures) or "no Gemini model configured")


def parse_json(raw: str) -> Any:
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        s = min([i for i in (raw.find("{"), raw.find("[")) if i >= 0] or [0])
        e = max(raw.rfind("}"), raw.rfind("]"))
        return json.loads(raw[s:e + 1])
