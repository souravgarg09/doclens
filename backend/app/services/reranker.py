"""Region relevance re-ranker: fuses SigLIP similarity with a Gemini listwise
relevance score (0-10) and light structural priors. Output is sorted, with a
`grounding` score in [0,1] used by the refusal guardrail."""
from typing import Any

from ..config import settings
from . import gemini


def _intent(question: str) -> dict[str, Any]:
    q = question.lower()
    hints: list[str] = []
    if "table" in q: hints.append("table")
    if any(w in q for w in ("chart", "plot", "graph", "trend", "curve", "axis")): hints += ["chart", "figure"]
    if any(w in q for w in ("figure", "fig.", "diagram")): hints += ["figure", "chart"]
    if any(w in q for w in ("equation", "formula", "loss", "objective")): hints.append("equation")
    wants_number = any(w in q for w in ("how many", "how much", "what was", "what is the", "%", "percent", "rate", "margin", "revenue", "accuracy", "increase", "decrease", "difference", "average", "total"))
    return {"type_hints": hints, "wants_number": wants_number}


def rerank(question: str, candidates: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], float, str]:
    if not candidates:
        return [], 0.0, "no candidates"
    intent = _intent(question)
    scored = [dict(candidate) for candidate in candidates]
    for c in scored:
        # Qdrant cosine scores are already comparable in [-1, 1]. Do not divide
        # by the best hit: that made an irrelevant best hit look perfectly grounded.
        s = max(0.0, min(1.0, float(c["score"])))
        if c["type"] in intent["type_hints"]:
            s += 0.08
        has_struct = bool((c.get("structured") or {}).get("cells") or (c.get("structured") or {}).get("series") or (c.get("structured") or {}).get("latex"))
        if intent["wants_number"] and has_struct:
            s += 0.05
        c["_prior"] = min(1.0, s)

    used = "SigLIP+priors"
    if settings.gemini_api_key:
        try:
            listing = "\n\n".join(f"[{c['region_id']}] ({c['type']}, page {c['page']})\n{(c.get('content') or '')[:900]}" for c in scored)
            raw = gemini.generate([f"Question: {question}\n\nScore each region 0-10 for how directly it answers the question. Return JSON {{\"scores\": {{\"<id>\": <0-10>}}}}.\n\n{listing}"])
            scores = gemini.parse_json(raw).get("scores", {})
            for c in scored:
                value = scores.get(str(c["region_id"]))
                if value is not None:
                    g = max(0.0, min(10.0, float(value))) / 10
                    c["_prior"] = 0.45 * c["_prior"] + 0.55 * g
            used = "SigLIP+Gemini listwise"
        except Exception as e:  # noqa: BLE001
            used = f"SigLIP+priors (gemini rerank failed: {e})"
    ranked = sorted(scored, key=lambda c: c["_prior"], reverse=True)
    for c in ranked:
        c["rerank_score"] = round(c.pop("_prior"), 4)
    kept = ranked[: settings.top_k_context]
    return kept, kept[0]["rerank_score"], f"{used}; kept {len(kept)}/{len(candidates)}"
