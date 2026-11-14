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
_POOL = ThreadPoolExecutor(max_workers=16, thread_name_prefix="gemini")  # abandoned slow requests finish here

# Model health: Gemini models get slow or unavailable at different times. A model that timed
# out or failed recently is not used as the hedged backup for a while.
_UNHEALTHY_FOR_S = 300.0
_last_failure: dict[str, float] = {}


def _record(model: str, future: Future) -> None:
    error = future.exception()
    if error is None:
        _last_failure.pop(model, None)
    elif _classify(error) != "permanent":
        _last_failure[model] = time.monotonic()


def _healthy(model: str) -> bool:
    failed_at = _last_failure.get(model)
    return failed_at is None or time.monotonic() - failed_at > _UNHEALTHY_FOR_S


def _submit(model: str, call: Callable[[], Any]) -> Future:
    future = _POOL.submit(call)
    future.add_done_callback(lambda f: _record(model, f))
    return future


class GeminiUnavailable(RuntimeError):
    """Raised when every configured model failed within the time budget."""


def client() -> genai.Client:
    global _client
    if _client is None:
        if not settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        # No client-level timeout: with it, the SDK silently retries a timed-out request, so a
        # "30 s" limit became 30 s + a hidden retry (requests ran 49-82 s). The limit is set per
        # request in generate() instead, where it is enforced exactly and retries are ours.
        # attempts=1: the SDK would otherwise retry 408/409/429/5xx up to 4 times with back-off
        # (up to 30 s) before we ever see the error, so 429s and 503s became silent slowness
        # (AI Studio's usage page showed them; our logs did not). Retries are handled here.
        _client = genai.Client(api_key=settings.gemini_api_key,
                               http_options=types.HttpOptions(retry_options=types.HttpRetryOptions(attempts=1)))
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
    if isinstance(e, _Deadline):
        return "transient"
    text = f"{type(e).__name__} {e}".lower()
    if code is None and any(word in text for word in _TRANSIENT_ERROR_WORDS):
        return "transient"
    return "permanent"


def _models() -> list[str]:
    """Primary model, then the fallbacks in order (GEMINI_FALLBACK_MODEL may be a comma list)."""
    models = [settings.gemini_model.strip()]
    for name in settings.gemini_fallback_model.split(","):
        name = name.strip()
        if name and name not in models:
            models.append(name)
    return models


class _Deadline(TimeoutError):
    """Neither the request nor its backup answered within the attempt's wall-clock limit."""


def _hedged(model: str, call: Callable[[], Any], backup_model: str,
            backup: Callable[[], Any], deadline: float) -> tuple[Any, bool]:
    """Run call(); if it is still running after the hedge delay, race `backup`.

    The backup goes to the next healthy model in the chain (see _backup_for): another
    model is served from separate capacity, so it often answers when this one is slow.
    Returns (response, answered_by_backup).
    """
    first = _submit(model, call)
    hedge_at = min(settings.gemini_hedge_after_s, max(0.0, deadline - time.monotonic()))
    if settings.gemini_hedge_after_s <= 0:
        done, _ = wait([first], timeout=max(0.0, deadline - time.monotonic()))
        if not done:
            raise _Deadline(f"{model}: no answer within the time limit")
        return first.result(), False
    done, _ = wait([first], timeout=hedge_at)
    if done:
        return first.result(), False
    log.info("generate: no reply after %.0fs; sending a hedged backup request", settings.gemini_hedge_after_s)
    second = _submit(backup_model, backup)
    pending: set[Future] = {first, second}
    error: BaseException | None = None
    while pending:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise _Deadline(f"{model}/{backup_model}: no answer within the time limit")
        finished, pending = wait(pending, timeout=remaining, return_when=FIRST_COMPLETED)
        for future in finished:
            if future.exception() is None:
                return future.result(), future is second  # the slower one finishes in the background
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
    chain = _models()
    extra: dict[str, Any] = {}
    if settings.gemini_thinking:
        extra["generation_config"] = {"thinking_level": settings.gemini_thinking}

    def request(name: str) -> Callable[[], Any]:
        return lambda: client().interactions.create(model=name, input=inputs, system_instruction=system,
                                                    response_format=response_format, store=False,
                                                    timeout=settings.gemini_timeout_s, **extra)

    def backup_for(index: int) -> str:
        """Next model in the chain that has not failed recently, else the same model again."""
        for candidate in chain[index + 1:]:
            if _healthy(candidate):
                return candidate
        return chain[index]

    for index, model in enumerate(chain):
        if not _healthy(model) and any(_healthy(m) for m in chain[index + 1:]):
            log.info("generate: skipping %s (failed in the last %.0f min)", model, _UNHEALTHY_FOR_S / 60)
            continue
        for attempt in range(1, settings.gemini_max_attempts + 1):
            backup_model = backup_for(index)
            if time.monotonic() - started > settings.gemini_budget_s:
                failures.append(f"{model}: time budget of {settings.gemini_budget_s:.0f}s used up")
                raise GeminiUnavailable("; ".join(failures))
            try:
                # wall-clock limit for this attempt (main + backup together), never past the budget
                deadline = min(time.monotonic() + settings.gemini_timeout_s, started + settings.gemini_budget_s)
                resp, by_backup = _hedged(model, request(model), backup_model, request(backup_model), deadline)
                answered_by = backup_model if by_backup else model
                log.info("generate: %s answered in %.1fs%s", answered_by, time.monotonic() - started,
                         " (hedged backup)" if by_backup else "" if model == settings.gemini_model else " (fallback)")
                return resp.output_text or ""
            except Exception as e:  # noqa: BLE001 - the SDK raises different classes per API surface
                kind = _classify(e)
                if kind == "permanent":
                    raise
                failures.append(f"{model} attempt {attempt}: {type(e).__name__} {str(e)[:160]}")
                if kind == "rate_limited":
                    log.warning("generate: %s is rate limited (429); switching model", model)
                    break
                if isinstance(e, _Deadline):
                    # this model (and its backup) is slow right now; retrying it rarely helps
                    log.warning("generate: %s did not answer within %.0fs; trying the next model",
                                model, settings.gemini_timeout_s)
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
