"""Orders the retrieved regions and scores how well the document can answer at all.

* Order: the fused retrieval rank (text vectors + BM25 + SigLIP for visuals), nudged toward
  the region types the question names ("table", "chart", ...) and toward structured regions
  when it asks for a number. An optional Gemini listwise score (settings.gemini_rerank)
  is blended in.
* `rerank_score` per region, and `grounding` = the best of them, are a calibrated relevance
  in [0, 1] from the text-embedder cosine plus keyword overlap. The refusal guardrail compares
  grounding with settings.grounding_threshold (0.35).

Calibration (arctic-embed-s, 30 on-topic and 8 off-topic questions over two papers): the best
region scored cosine >= 0.587 for every on-topic question and <= 0.583 for off-topic ones
("capital of France", "price of Bitcoin"). relevance = (cosine - 0.45) / 0.30 puts the 0.35
threshold at cosine 0.555: most off-topic questions are refused without a Gemini call, and the
model's own "not in the regions" refusal catches the rest. SigLIP's cosine could not tell the
two apart (off-topic up to 0.94, on-topic from 0.89), so it is not used for grounding.
"""
from typing import Any

from ..config import settings
from . import embeddings, gemini

_FLOOR, _SPAN = 0.45, 0.30


def _intent(question: str) -> dict[str, Any]:
    q = question.lower()
    hints: list[str] = []
    if "table" in q: hints.append("table")
    if any(w in q for w in ("chart", "plot", "graph", "trend", "curve", "axis")): hints += ["chart", "figure"]
    if any(w in q for w in ("figure", "fig.", "diagram")): hints += ["figure", "chart"]
    if any(w in q for w in ("equation", "formula", "loss", "objective")): hints.append("equation")
    wants_number = any(w in q for w in ("how many", "how much", "what was", "what is the", "%", "percent", "rate", "margin", "revenue", "accuracy", "increase", "decrease", "difference", "average", "total"))
    return {"type_hints": hints, "wants_number": wants_number}


def relevance(cosine: float) -> float:
    """Text-embedder cosine -> calibrated relevance in [0, 1] (see module docstring)."""
    if not embeddings.text_enabled():
        return max(0.0, min(1.0, cosine))  # SigLIP-only fallback: the old uncalibrated score
    return max(0.0, min(1.0, (cosine - _FLOOR) / _SPAN))


def rerank(question: str, candidates: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], float, str]:
    if not candidates:
        return [], 0.0, "no candidates"
    intent = _intent(question)
    scored = [dict(candidate) for candidate in candidates]
    top_lexical = max((float(c.get("lexical") or 0.0) for c in scored), default=0.0)
    top_fused = max((float(c.get("fused") or 0.0) for c in scored), default=0.0) or 1.0
    for c in scored:
        s = c.get("structured") or {}
        has_struct = bool(s.get("cells") or s.get("series") or s.get("latex"))
        order = float(c.get("fused") or 0.0) / top_fused
        if c["type"] in intent["type_hints"]:
            order += 0.15
        if intent["wants_number"] and has_struct:
            order += 0.05
        c["_order"] = order
        lexical_share = float(c.get("lexical") or 0.0) / top_lexical if top_lexical > 0 else 0.0
        c["_relevance"] = min(1.0, relevance(float(c["score"])) + 0.15 * lexical_share)

    used = "fused text+BM25+visual rank" if embeddings.text_enabled() else "fused SigLIP+BM25 rank"
    if settings.gemini_api_key and settings.gemini_rerank:
        try:
            listing = "\n\n".join(f"[{c['region_id']}] ({c['type']}, page {c['page']})\n{(c.get('content') or '')[:900]}" for c in scored)
            raw = gemini.generate([f"Question: {question}\n\nScore each region 0-10 for how directly it answers the question. Return JSON {{\"scores\": {{\"<id>\": <0-10>}}}}.\n\n{listing}"])
            scores = gemini.parse_json(raw).get("scores", {})
            for c in scored:
                value = scores.get(str(c["region_id"]))
                if value is not None:
                    g = max(0.0, min(10.0, float(value))) / 10
                    c["_order"] = 0.45 * c["_order"] + 0.55 * g
                    c["_relevance"] = max(c["_relevance"], g)
            used += " + Gemini listwise"
        except Exception as e:  # noqa: BLE001
            used += f" (gemini rerank failed: {e})"
    ranked = sorted(scored, key=lambda c: c["_order"], reverse=True)
    for c in ranked:
        c.pop("_order")
        c["rerank_score"] = round(c.pop("_relevance"), 4)
    kept = ranked[: settings.top_k_context]
    grounding = max(c["rerank_score"] for c in kept)
    return kept, grounding, f"{used}; kept {len(kept)}/{len(candidates)}"
