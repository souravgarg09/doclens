"""LangGraph orchestration of the question-answering agent.

    parse ─▶ retrieve ─▶ rerank ─▶ expand ─▶ (guardrail) ─▶ answer ─▶ verify ─▶ END
                                                 │                       │
                                                 └─▶ refuse ─▶ END       └─▶ answer (repair, ≤ MAX_VERIFY_ATTEMPTS)

expand: makes sure the VLM actually sees the table/figure the question is about.
  1. Explicit label ("Figure 2", "Table 1") → look up that caption in Postgres, pull the
     structured regions from that page, matching type first.
  2. Otherwise, caption-like regions already in context → pull structured siblings from those pages.
  3. Otherwise, question names a table/figure but context has none → siblings from top pages.
"""
from __future__ import annotations

import asyncio
import re
import time
from typing import Any, Literal, TypedDict

from langgraph.graph import END, StateGraph
from sqlalchemy import select

from ..config import APP_NAME, settings
from ..db import Document, Query, Region, SessionLocal
from ..services import gemini
from ..services.gemini import GeminiUnavailable
from ..services.reranker import rerank as rerank_regions
from ..services.retriever import retrieve as retrieve_regions
from ..services.verifier import (
    extract_region_citation_ids,
    normalize_region_citations,
    uncited_factual_sentences,
    verify as verify_numbers,
)


class AgentState(TypedDict, total=False):
    document_id: int
    question: str
    doc_status: str
    candidates: list[dict[str, Any]]
    context: list[dict[str, Any]]
    grounding: float
    answer: str
    cited_ids: list[int]
    verification: dict[str, Any]
    feedback: str
    attempts: int
    error: str
    status: Literal["answered", "refused", "flagged"]
    trace: list[dict[str, Any]]


# ── constants ────────────────────────────────────────────────────────────────
_STRUCTURED = ("table", "chart", "figure", "equation")

# caption-like region text: "TABLE 1. ...", "Fig. 2 ...", "Figure 3: ...", "Equation 4 ..."
_CAPTION_RE = re.compile(r"^\s*(table|fig(?:ure)?|eq(?:uation)?)\s*\.?\s*\d+", re.I)

# words in the question that imply a structured region
_ASKS_STRUCTURED_RE = re.compile(r"\b(table|figure|fig|chart|plot|graph|equation|formula)\b", re.I)

# explicit numbered references in the question: "Table 1", "Fig. 2", "Figure 3", "Equation 4"
_LABEL_REF_RE = re.compile(r"\b(table|fig(?:ure)?|eq(?:uation)?)\s*\.?\s*(\d+)\b", re.I)

# regex fragment used to find the caption for a given label kind
_KIND_PATTERN = {
    "table": r"table",
    "fig": r"fig(?:ure)?",
    "figure": r"fig(?:ure)?",
    "eq": r"eq(?:uation)?",
    "equation": r"eq(?:uation)?",
}

# which region types a label kind refers to
_KIND_TYPES = {
    "table": ("table",),
    "fig": ("figure", "chart"),
    "figure": ("figure", "chart"),
    "eq": ("equation",),
    "equation": ("equation",),
}

# which region types a plain word in the question refers to
_WORD_TYPES = {
    "table": ("table",),
    "figure": ("figure", "chart"),
    "fig": ("figure", "chart"),
    "chart": ("chart", "figure"),
    "plot": ("chart", "figure"),
    "graph": ("chart", "figure"),
    "equation": ("equation",),
    "formula": ("equation",),
}

_MAX_SIBLINGS = 6
_LABEL_MATCH_GROUNDING = 0.6  # an exact caption hit is strong evidence


def _t(state: AgentState, node: str, detail: str, t0: float) -> list[dict[str, Any]]:
    return [*state.get("trace", []), {"node": node, "detail": detail, "ms": int((time.time() - t0) * 1000)}]


def _wanted_types(question: str) -> tuple[str, ...]:
    """Structured types the question is asking about (empty = no preference)."""
    out: list[str] = []
    for w in _ASKS_STRUCTURED_RE.findall(question):
        for t in _WORD_TYPES.get(w.lower(), ()):
            if t not in out:
                out.append(t)
    return tuple(out)


# ── nodes ────────────────────────────────────────────────────────────────────
async def parse_node(state: AgentState) -> AgentState:
    t0 = time.time()
    async with SessionLocal() as s:
        doc = await s.get(Document, state["document_id"])
        n = len((await s.execute(select(Region.id).where(Region.document_id == state["document_id"]))).all()) if doc else 0
    status = doc.status if doc else "missing"
    return {"doc_status": status, "attempts": 0, "error": "",
            "trace": _t(state, "parse", f"document status={status}, {n} regions indexed", t0)}


async def retrieve_node(state: AgentState) -> AgentState:
    t0 = time.time()
    async with SessionLocal() as s:
        cands = await retrieve_regions(s, state["document_id"], state["question"])
    keyword_hits = sum(1 for c in cands if c.get("lexical"))
    return {"candidates": cands, "trace": _t(state, "retrieve",
                                             f"hybrid SigLIP+BM25 returned {len(cands)} candidates "
                                             f"({keyword_hits} with keyword matches)", t0)}


async def rerank_node(state: AgentState) -> AgentState:
    t0 = time.time()
    ctx, grounding, detail = await asyncio.to_thread(rerank_regions, state["question"], state["candidates"])
    return {"context": ctx, "grounding": grounding, "trace": _t(state, "rerank", f"{detail}; grounding={grounding:.2f}", t0)}


async def expand_node(state: AgentState) -> AgentState:
    t0 = time.time()
    ctx = list(state.get("context") or [])
    q = state["question"]
    doc_id = state["document_id"]
    have = {c["id"] for c in ctx}
    wanted = _wanted_types(q)
    grounding = float(state.get("grounding", 0.0))
    how = ""
    pages: set[int] = set()

    async with SessionLocal() as s:
        # ── 1. explicit label lookup ("Figure 2", "Table 1") ──
        refs = [(k.lower(), int(n)) for k, n in _LABEL_REF_RE.findall(q)]
        if refs:
            caps = (await s.execute(
                select(Region.id, Region.page, Region.content)
                .where(Region.document_id == doc_id, Region.type.in_(("text", "title")))
            )).all()
            for kind, num in refs:
                pat = re.compile(rf"^\s*{_KIND_PATTERN.get(kind, kind)}\s*\.?\s*{num}\b", re.I)
                for cid, page, content in caps:
                    if pat.match(content or ""):
                        pages.add(page)
                        how = f"label '{kind} {num}' → caption R{cid}@p{page}"
                        for t in _KIND_TYPES.get(kind, ()):
                            if t not in wanted:
                                wanted = (*wanted, t)
            if pages:
                grounding = max(grounding, _LABEL_MATCH_GROUNDING)

        # ── 2 / 3. fallbacks ──
        if not pages:
            caption_pages = {c["page"] for c in ctx
                             if c.get("type") in ("text", "title") and _CAPTION_RE.match(c.get("content") or "")}
            has_structured = any(c.get("type") in _STRUCTURED for c in ctx)
            if caption_pages:
                pages, how = caption_pages, "caption regions in context"
            elif wanted and not has_structured and ctx:
                pages, how = {c["page"] for c in ctx[:3]}, "question names a structured type; none in context"

        if not pages:
            return {"context": ctx, "grounding": grounding,
                    "trace": _t(state, "expand", "no label/caption hint; context unchanged", t0)}

        rows = (await s.execute(
            select(Region)
            .where(Region.document_id == doc_id,
                   Region.page.in_(sorted(pages)),
                   Region.type.in_(_STRUCTURED))
        )).scalars().all()

    # matching types first, then page, then id — so the cap never starves the type we want
    rows.sort(key=lambda r: (0 if r.type in wanted else 1, r.page, r.id))
    added: list[str] = []
    for r in rows:
        if r.id in have:
            continue
        d = r.as_dict(include_paths=True)
        d["rerank_score"] = 0.0  # keep numeric for the frontend
        d["via"] = "sibling"
        ctx.append(d)
        have.add(r.id)
        added.append(f"R{r.id}:{r.type}@p{r.page}")
        if len(added) >= _MAX_SIBLINGS:
            break

    return {"context": ctx, "grounding": grounding,
            "trace": _t(state, "expand",
                        f"{how}; pages={sorted(pages)} want={list(wanted) or 'any'} "
                        f"added {' '.join(added) if added else 'nothing'}", t0)}


async def refuse_node(state: AgentState) -> AgentState:
    t0 = time.time()
    g = state.get("grounding", 0.0)
    if state.get("doc_status") and state["doc_status"] != "ready":
        msg = f"This document is not ready yet (status={state['doc_status']}). Wait for parsing to finish, then ask again."
        return {"answer": msg, "cited_ids": [], "status": "refused",
                "trace": _t(state, "guardrail", f"REFUSED: document status={state['doc_status']}", t0)}
    msg = (f"I can't answer this from the document: no region is grounded strongly enough "
           f"(best grounding {g:.2f} < {settings.grounding_threshold}). Ask about a specific table, figure or metric in the paper.")
    return {"answer": msg, "cited_ids": [], "status": "refused", "trace": _t(state, "guardrail", f"REFUSED grounding {g:.2f}", t0)}


SYSTEM = f"""You are {APP_NAME}, a document-intelligence reasoner. Answer ONLY from the provided regions.
Tables, charts and equations are given BOTH as extracted text AND as images. If the extracted text is
incomplete or garbled, READ THE IMAGE — it is the ground truth. Do not refuse if the value is visible in the image.
Every factual sentence must end with citations like [R123] or [R123] [R456] using the region ids.
Never combine multiple ids inside one bracket. When reading tables/charts,
quote the exact cell/point values you use and show any arithmetic.
Only if none of the regions (text or images) contain the answer, return {{"answer": "", "cited_region_ids": [], "confidence": 0}}.
Return JSON: {{"answer": string, "cited_region_ids": number[], "confidence": number 0-1}}."""

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "cited_region_ids": {"type": "array", "items": {"type": "integer"}},
        "confidence": {"type": "number"},
    },
    "required": ["answer", "cited_region_ids", "confidence"],
}


def _build_parts(state: AgentState) -> tuple[list[Any], list[str]]:
    """Assemble the multimodal prompt. Returns (parts, per-region summary for the trace)."""
    parts: list[Any] = []
    seen: list[str] = []
    for c in state["context"]:
        s = c.get("structured") or {}
        extra = ""
        if s.get("cells"):
            extra = "\nCELLS: " + " ; ".join(f"[{x['row']},{x['col']}] {x['text']}" for x in s["cells"][:400])
        elif s.get("series"):
            extra = ("\nSERIES: " + " | ".join(
                f"{ser['name']}: " + ", ".join(f"{p['x']}={p['y']}" for p in ser["points"]) for ser in s["series"])
                + f"\nAXES: {s.get('axes')}")
        elif s.get("latex"):
            extra = f"\nLATEX: {s['latex']}"
        content = c.get("content") or ""
        parts.append(f"[R{c['id']}] type={c['type']} page={c['page']} relevance={c.get('rerank_score')}\n{content[:1500]}{extra}")
        has_img = False
        if c.get("crop_path") and c["type"] in _STRUCTURED:
            try:
                parts.append(gemini.image_part(c["crop_path"]))  # let the VLM read the pixels
                has_img = True
            except OSError:
                pass
        seen.append(f"R{c['id']}:{c['type']}@p{c['page']}({len(content)}ch{',img' if has_img else ''})")
    q = f"QUESTION: {state['question']}"
    if state.get("feedback"):
        q += f"\n\nVERIFIER FEEDBACK - correct the answer using only the cited evidence: {state['feedback']}"
    parts.append(q)
    return parts, seen


async def answer_node(state: AgentState) -> AgentState:
    t0 = time.time()
    attempt = state.get("attempts", 0) + 1
    parts, seen = _build_parts(state)

    # 1. call the model — distinguish outage from a genuine "not in document"
    try:
        raw = await asyncio.to_thread(gemini.generate, parts, system=SYSTEM, schema=ANSWER_SCHEMA)
    except GeminiUnavailable as e:
        return {"answer": "", "cited_ids": [], "attempts": attempt, "error": str(e),
                "trace": _t(state, "answer", f"attempt {attempt}: MODEL UNAVAILABLE — {e}", t0)}

    # 2. parse — tolerate junk output instead of crashing the whole request
    try:
        data = gemini.parse_json(raw)
    except Exception as e:  # noqa: BLE001
        return {"answer": "", "cited_ids": [], "attempts": attempt, "error": f"unparseable model output: {e}",
                "trace": _t(state, "answer", f"attempt {attempt}: UNPARSEABLE raw={raw[:300]!r}", t0)}
    if isinstance(data, list):
        data = data[0] if data and isinstance(data[0], dict) else {}
    if not isinstance(data, dict):
        data = {}

    # 3. extract answer + citations
    answer = next((data.get(k) for k in ("answer", "response", "final_answer", "text") if data.get(k)), "") or ""
    answer = normalize_region_citations(str(answer))
    valid = {c["id"] for c in state["context"]}
    cited = {int(i) for i in (data.get("cited_region_ids") or []) if str(i).isdigit()}
    cited |= set(extract_region_citation_ids(answer))
    cited_sorted = sorted(cited & valid)

    detail = (f"attempt {attempt}: {len(answer)} chars, {len(cited_sorted)} citations, conf={data.get('confidence')}"
              f" | ctx={' '.join(seen)}")
    if not answer:
        detail += f" | raw={raw[:300]!r}"  # show exactly what the model said when it refused
    return {"answer": answer, "cited_ids": cited_sorted, "attempts": attempt, "error": "",
            "trace": _t(state, "answer", detail, t0)}


async def verify_node(state: AgentState) -> AgentState:
    t0 = time.time()
    if state.get("error"):
        return {"status": "refused", "cited_ids": [],
                "answer": "The model is temporarily unavailable or returned malformed output. Please try again in a moment.",
                "trace": _t(state, "guardrail", f"REFUSED: {state['error'][:200]}", t0)}
    if not state.get("answer") or not state.get("cited_ids"):
        return {"status": "refused", "cited_ids": [],
                "answer": state.get("answer") or "The cited regions do not contain enough evidence; refusing rather than guessing.",
                "trace": _t(state, "guardrail", "REFUSED: model found no answer in the cited regions", t0)}
    cited = [c for c in state["context"] if c["id"] in state["cited_ids"]]
    v = verify_numbers(state["answer"], cited).to_dict()
    feedback = [f"claim {c['value']} unsupported (nearest {c['evidence']} = {c['recomputed']})"
                for c in v["checks"] if not c["matched"]]
    uncited = uncited_factual_sentences(state["answer"])
    if uncited:
        feedback.append(f"add a valid [R id] citation at the end of {len(uncited)} factual sentence(s)")
        v["status"] = "citation_gap"
        v["summary"] = f"{len(uncited)} factual sentence(s) do not end with a region citation. " + v["summary"]
    fb = "; ".join(feedback)
    status = "flagged" if v["status"] in ("contradiction", "unverifiable", "citation_gap") else "answered"
    return {"verification": v, "feedback": fb, "status": status,
            "trace": _t(state, "verify", f"{v['status']}: {v['summary']}", t0)}


# ── edges ────────────────────────────────────────────────────────────────────
def after_parse(state: AgentState) -> str:
    return "retrieve" if state["doc_status"] == "ready" else "refuse"


def guardrail(state: AgentState) -> str:
    return "answer" if state.get("context") and state.get("grounding", 0.0) >= settings.grounding_threshold else "refuse"


def after_verify(state: AgentState) -> str:
    if state.get("status") == "flagged" and state.get("attempts", 0) < settings.max_verify_attempts:
        return "answer"  # one repair pass with verifier feedback
    return END


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("parse", parse_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("rerank", rerank_node)
    g.add_node("expand", expand_node)
    g.add_node("refuse", refuse_node)
    g.add_node("generate_answer", answer_node)
    g.add_node("verify", verify_node)
    g.set_entry_point("parse")
    g.add_conditional_edges("parse", after_parse, {"retrieve": "retrieve", "refuse": "refuse"})
    g.add_edge("retrieve", "rerank")
    g.add_edge("rerank", "expand")
    g.add_conditional_edges("expand", guardrail, {"answer": "generate_answer", "refuse": "refuse"})
    g.add_edge("generate_answer", "verify")
    g.add_conditional_edges("verify", after_verify, {"answer": "generate_answer", END: END})
    g.add_edge("refuse", END)
    return g.compile()


GRAPH = build_graph()


async def run_agent(document_id: int, question: str) -> dict[str, Any]:
    final: AgentState = await GRAPH.ainvoke({"document_id": document_id, "question": question, "trace": []})
    ctx = final.get("context", [])
    cited_ids = set(final.get("cited_ids", []))
    citations = [{"region_id": c["id"], "page": c["page"], "type": c["type"], "bbox": c["bbox"],
                  "score": c.get("rerank_score", 0), "snippet": (c.get("content") or "")[:220]}
                 for c in ctx if c["id"] in cited_ids]
    async with SessionLocal() as s:
        row = Query(document_id=document_id, question=question, answer=final.get("answer", ""),
                    status=final.get("status", "refused"), grounding=float(final.get("grounding", 0.0)),
                    citations=citations, verification=final.get("verification"), trace=final.get("trace", []))
        s.add(row)
        await s.commit()
        await s.refresh(row)
        return {"id": row.id, "document_id": document_id, "question": question, "answer": row.answer,
                "status": row.status, "grounding": row.grounding, "citations": citations,
                "verification": row.verification, "trace": row.trace}
