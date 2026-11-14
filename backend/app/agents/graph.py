"""LangGraph orchestration of the question-answering agent.

    parse ─▶ retrieve ─▶ rerank ─▶ expand ─▶ links ─▶ (guardrail) ─▶ answer ─▶ verify ─▶ END
                                                           │                       │
                                                           └─▶ refuse ─▶ END       └─▶ answer (repair, ≤ MAX_VERIFY_ATTEMPTS)

expand: makes sure the VLM actually sees the table/figure the question is about.
  1. Explicit label ("Figure 2", "Table 1") → look up that caption in Postgres, pull the
     structured regions from that page, matching type first.
  2. Otherwise, caption-like regions already in context → pull structured siblings from those pages.
  3. Otherwise, question names a table/figure but context has none → siblings from top pages.

links: follows the PDF's own hyperlinks (see workers/parsers/links.py).
  1. "reference 11" / "[11]" in the question → the reference-list entry and the passages citing it.
  2. the question asks for code / a dataset / a website → regions holding URL/DOI links are added.
  3. a context region links to a table/figure/equation ("see Table 2") → that region is added.
  4. every context region is shown to the model with its links: citation → reference text,
     words → URL/DOI, "Table 2" → caption.
"""
from __future__ import annotations

import asyncio
import re
import time
from typing import Any, Literal, TypedDict

from langgraph.graph import END, StateGraph
from sqlalchemy import select

from ..config import APP_NAME, settings
from ..db import Document, Link, Query, Region, SessionLocal
from ..services import gemini, lexical
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

# "reference 11", "ref. [11]", "citation 11", "[11]", "bibliography entry 11"
_REFERENCE_RE = re.compile(
    r"\b(?:ref(?:erence)?s?|citations?|cited\s+(?:work|paper|study|article)|"
    r"(?:reference|bibliography)\s+(?:entry|item|number))\.?\s*(?:no\.?|number|#)?\s*[\[(]?(\d{1,3})[\])]?"
    r"|\[(\d{1,3})\]", re.I)
# questions that ask for a web address: the regions holding URL/DOI links are added
_WANTS_URL_RE = re.compile(
    r"\b(code|source code|github|gitlab|repo(?:sitory)?|url|link|website|web ?site|homepage|download|"
    r"available|doi|dataset|data set|implementation|demo|project page)\b", re.I)
_MAX_URL_REGIONS = 2
_MAX_LINK_FOLLOWS = 3
_MAX_LINKS_PER_REGION = 10
_LINK_GROUNDING = 0.6
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
                                             f"hybrid text+BM25+visual returned {len(cands)} candidates "
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


def _link_line(link: Link, region_page: dict[int, int]) -> str:
    """One LINKS entry for the prompt: what the reader sees -> where it points."""
    shown = (link.text or "").strip()[:40] or link.kind
    if link.uri:
        return f"'{shown}' -> {link.kind} {link.uri[:160]}"
    target = re.sub(r"\s+", " ", link.target_text or "").strip()
    where = f"p{link.target_page}" if link.target_page else ""
    if link.target_region_id and link.target_region_id in region_page:
        where += f", R{link.target_region_id}"
    label = f"reference [{link.label}]" if link.kind == "citation" and link.label else link.kind
    return f"'{shown}' -> {label}: {target[:170]} ({where})" if target else f"'{shown}' -> {label} ({where})"


async def links_node(state: AgentState) -> AgentState:
    t0 = time.time()
    ctx = list(state.get("context") or [])
    doc_id = state["document_id"]
    grounding = float(state.get("grounding", 0.0))
    have = {c["id"] for c in ctx}
    notes: dict[int, list[str]] = {}
    added: list[str] = []
    how: list[str] = []

    async with SessionLocal() as s:
        rows = (await s.execute(select(Link).where(Link.document_id == doc_id))).scalars().all()
        if not rows:
            return {"trace": _t(state, "links", "this PDF has no embedded links or numbered reference list", t0)}
        want: dict[int, str] = {}  # region id -> why it is added

        # 1. "reference 11" in the question
        numbers = [a or b for a, b in _REFERENCE_RE.findall(state["question"])]
        for number in dict.fromkeys(numbers):
            entry = next((r for r in rows if r.kind == "reference" and r.label == number), None)
            citing = [r for r in rows if r.kind == "citation" and r.label == number]
            text = entry.text if entry else next((r.target_text for r in citing if r.target_text), "")
            holder = (entry.region_id if entry else None) or next(
                (r.target_region_id for r in citing if r.target_region_id), None)
            if not text or not holder:
                how.append(f"reference [{number}] not found")
                continue
            want.setdefault(holder, f"reference [{number}]")
            notes.setdefault(holder, []).append(f"REFERENCE [{number}] (from the reference list): {text}")
            for r in citing[:2]:
                if r.region_id:
                    want.setdefault(r.region_id, f"cites [{number}]")
            grounding = max(grounding, _LINK_GROUNDING)
            how.append(f"reference [{number}] -> R{holder} + {min(2, len(citing))} citing passage(s)")

        # 2. "where is the code?": regions whose words link to a URL / DOI, best keyword match first
        if _WANTS_URL_RE.search(state["question"]):
            q_terms = set(lexical.tokenize(state["question"]))
            holders: dict[int, int] = {}
            for r in rows:
                if r.kind in ("url", "doi") and r.region_id:
                    overlap = len(q_terms & set(lexical.tokenize(f"{r.text} {r.uri}")))
                    holders[r.region_id] = max(holders.get(r.region_id, 0), overlap)
            if holders:
                texts = {rid: content for rid, content in (await s.execute(
                    select(Region.id, Region.content).where(Region.id.in_(list(holders))))).all()}
                for rid in holders:
                    holders[rid] += len(q_terms & set(lexical.tokenize(texts.get(rid) or "")))
                picked = [rid for rid in sorted(holders, key=holders.get, reverse=True)
                          if holders[rid] > 0 and rid not in have][:_MAX_URL_REGIONS]
                for rid in picked:
                    want.setdefault(rid, "holds a URL/DOI link")
                if picked:
                    how.append(f"question asks for a web address -> {len(picked)} region(s) with URL/DOI links")

        # 3. links from the context to tables / figures / equations
        follows = 0
        for r in rows:
            if (r.region_id in have and r.kind in ("table", "figure", "equation") and r.target_region_id
                    and r.target_region_id not in have and r.target_region_id not in want
                    and follows < _MAX_LINK_FOLLOWS):
                want[r.target_region_id] = f"link '{r.text}' from R{r.region_id}"
                follows += 1

        new_ids = [i for i in want if i not in have]
        if new_ids:
            for region in (await s.execute(select(Region).where(Region.id.in_(new_ids)))).scalars().all():
                d = region.as_dict(include_paths=True)
                d["rerank_score"] = 0.0
                d["via"] = "link"
                ctx.append(d)
                have.add(region.id)
                added.append(f"R{region.id}({want[region.id]})")

    # 4. show every context region's own links
    region_page = {c["id"]: c["page"] for c in ctx}
    by_region: dict[int, list[Link]] = {}
    for r in rows:
        if r.kind != "reference" and r.region_id in have:
            by_region.setdefault(r.region_id, []).append(r)
    shown = 0
    for c in ctx:
        lines = list(notes.get(c["id"], []))
        seen: set[str] = set()
        for link in by_region.get(c["id"], []):
            line = _link_line(link, region_page)
            if line not in seen and len(seen) < _MAX_LINKS_PER_REGION:
                seen.add(line)
                lines.append(line)
        if lines:
            c["links"] = lines
            shown += len(lines)
    detail = "; ".join(how + ([f"followed {' '.join(added)}"] if added else [])) or "no reference or link to follow"
    return {"context": ctx, "grounding": grounding,
            "trace": _t(state, "links", f"{detail}; {shown} link line(s) shown to the model ({len(rows)} in the PDF)", t0)}


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
Never combine multiple ids inside one bracket. A region may list LINKS: the PDF's own hyperlinks
inside it (a citation number -> the full reference entry, a word -> its URL or DOI, "Table 2" -> that
table). Use them to say which work a citation refers to, citing the region that holds the link. When reading tables/charts,
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
        if c.get("links"):
            extra += "\nLINKS: " + " ; ".join(c["links"])
        content = c.get("content") or ""
        parts.append(f"[R{c['id']}] type={c['type']} page={c['page']} relevance={c.get('rerank_score')}\n{content[:1500]}{extra}")
        has_img = False
        if c.get("crop_path") and c["type"] in _STRUCTURED:
            try:
                parts.append(gemini.image_part(c["crop_path"]))  # let the VLM read the pixels
                has_img = True
            except OSError:
                pass
        seen.append(f"R{c['id']}:{c['type']}@p{c['page']}({len(content)}ch{',img' if has_img else ''}"
                    f"{',' + str(len(c['links'])) + 'links' if c.get('links') else ''})")
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
    g.add_node("links", links_node)
    g.add_node("refuse", refuse_node)
    g.add_node("generate_answer", answer_node)
    g.add_node("verify", verify_node)
    g.set_entry_point("parse")
    g.add_conditional_edges("parse", after_parse, {"retrieve": "retrieve", "refuse": "refuse"})
    g.add_edge("retrieve", "rerank")
    g.add_edge("rerank", "expand")
    g.add_edge("expand", "links")
    g.add_conditional_edges("links", guardrail, {"answer": "generate_answer", "refuse": "refuse"})
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
