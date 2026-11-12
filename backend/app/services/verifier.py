"""Numeric-consistency verifier.

Extracts every numeric claim from the VLM answer and checks it against the
structured values of the cited regions (table cells, chart series points,
equation constants). Derived quantities (differences, ratios, % change, sums,
means) are *recomputed* from the evidence so that a textual claim contradicting
the figure/table is flagged instead of silently accepted.

Noise control (what is NOT treated as a numeric claim):
  • citations / region refs:    [R52]  (31)  (11, 26)
  • label references:           Figure 1, Table 2, Eq. 3, page 4, study 5
  • range / category labels:    2-3 sets, 4–6 sets, 19-44 y
  • ordinals, years, list numbering:  1st, 2002, "1. ..."
  • small bare integers (0-12) that are not written as a value ("= 3", "n = 9" are kept)

Precision-aware tolerance: a claim written as 0.50 must match to ±0.005; a claim of
1,234 matches to ±0.5. Derived (recomputed) matches are only allowed for non-trivial
claims so the verifier can't "invent" arithmetic to justify a stray 1 or 100.
"""
from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field, asdict
from typing import Any

NUM_RE = re.compile(r"(?<![\w.])[-+]?\$?\d{1,3}(?:,\d{3})+(?:\.\d+)?%?|(?<![\w.])[-+]?\$?\d+(?:\.\d+)?%?(?![\w])")

# ── noise patterns stripped before claim extraction ─────────────────────────
_CITATION_RE = re.compile(r"\[\s*R?\d+(?:\s*[,;]\s*R?\d+)*\s*\]|\(\s*R?\d+(?:\s*[,;–-]\s*R?\d+)*\s*\)")
_LABEL_REF_RE = re.compile(
    r"\b(?:table|tab|fig(?:ure)?s?|eq(?:uation)?s?|section|sec|pages?|p|chapter|appendix|panel|"
    r"model|study|studies|ref(?:erence)?s?|region|row|column|col|item|step|group)\.?\s*R?\d+[a-z]?\b", re.I)
_RANGE_LABEL_RE = re.compile(r"(?<![\d.])\d+\s*(?:-|–|—|to)\s*\d+(?![\d.%])", re.I)  # "2-3 sets", "19–44 y"
_ORDINAL_RE = re.compile(r"\b\d+(?:st|nd|rd|th)\b", re.I)
_YEAR_RE = re.compile(r"(?<![\d.$])(?:19|20)\d{2}(?!\d|\.\d)")
_LIST_NUM_RE = re.compile(r"(?m)^\s*\d+[.)]\s+")
_SMALL_INT_MAX = 12
_REGION_CITATION_GROUP_RE = re.compile(
    r"\[\s*R\d+(?:\s*[,;]\s*R?\d+)*\s*\]", re.I
)
_REGION_CITATION_AT_END_RE = re.compile(
    r"\[\s*R\d+(?:\s*[,;]\s*R?\d+)*\s*\][.!?]?$", re.I
)


def extract_region_citation_ids(text: str) -> list[int]:
    """Extract IDs from both ``[R1] [R2]`` and ``[R1, R2]`` forms."""
    ids: list[int] = []
    for group in _REGION_CITATION_GROUP_RE.findall(text or ""):
        ids.extend(int(value) for value in re.findall(r"(?:R\s*)?(\d+)", group, re.I))
    return ids


def normalize_region_citations(text: str) -> str:
    """Convert grouped citations to the canonical separately-clickable form."""
    def replace(match: re.Match[str]) -> str:
        return " ".join(f"[R{value}]" for value in extract_region_citation_ids(match.group(0)))

    return _REGION_CITATION_GROUP_RE.sub(replace, text or "")


def uncited_factual_sentences(answer: str) -> list[str]:
    """Return prose claims that lack a trailing region citation.

    Markdown headings and introductory labels such as ``Main findings:`` are
    presentation structure, not factual claims. Treating them as claims caused
    valid answers to enter an unnecessary repair pass and consume API quota.
    """
    uncited: list[str] = []
    for block in re.split(r"\n+", answer or ""):
        block = block.strip()
        if not block:
            continue
        if block.startswith("#") or block.endswith(":"):
            continue
        if re.fullmatch(r"\*\*[^*]+\*\*", block) or re.fullmatch(r"__[^_]+__", block):
            continue
        block = re.sub(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)", "", block)
        for sentence in re.split(r"(?<=[.!?])\s+", block):
            sentence = sentence.strip()
            if len(sentence.split()) < 4:
                continue
            if not _REGION_CITATION_AT_END_RE.search(sentence):
                uncited.append(sentence[:120])
    return uncited


def _tokens(text: str) -> list[tuple[float, int, str, int]]:
    """(value, decimals, raw, start) for every number in text."""
    out = []
    for m in NUM_RE.finditer(text or ""):
        raw = m.group(0)
        clean = raw.replace("$", "").replace(",", "").replace("%", "")
        try:
            v = float(clean)
        except ValueError:
            continue
        dec = len(clean.split(".")[1]) if "." in clean else 0
        out.append((v, dec, raw, m.start()))
    return out


def extract_numbers(text: str) -> list[float]:
    return [v for v, _, _, _ in _tokens(text)]


def _strip_noise(sent: str) -> str:
    s = _CITATION_RE.sub(" ", sent)
    s = _LABEL_REF_RE.sub(" ", s)
    s = _RANGE_LABEL_RE.sub(" ", s)
    s = _ORDINAL_RE.sub(" ", s)
    s = _YEAR_RE.sub(" ", s)
    s = _LIST_NUM_RE.sub(" ", s)
    return s


def _is_weak(v: float, dec: int, raw: str, cleaned: str, start: int) -> bool:
    """Small bare integer that looks like a count/label rather than a measured value."""
    if dec > 0 or "%" in raw or "$" in raw or not (0 <= v <= _SMALL_INT_MAX):
        return False
    pre = cleaned[max(0, start - 4):start]
    return not any(ch in pre for ch in "=:")  # "= 3", "n: 9" are explicit values → keep


def _numeric_claims(answer: str) -> tuple[list[tuple[str, float, int]], int]:
    """Returns (claims as (sentence, value, decimals), number of ignored tokens)."""
    claims: list[tuple[str, float, int]] = []
    ignored = 0
    for sent in re.split(r"(?<=[.!?])\s+|\n+", answer or ""):
        if not sent.strip():
            continue
        cleaned = _strip_noise(sent)
        ignored += len(_tokens(sent)) - len(_tokens(cleaned))
        for v, dec, raw, start in _tokens(cleaned):
            if _is_weak(v, dec, raw, cleaned, start):
                ignored += 1
                continue
            claims.append((sent.strip()[:200], v, dec))
    return claims, ignored


def numeric_claims(answer: str) -> list[tuple[str, float]]:
    """Backward-compatible view: (sentence, value)."""
    return [(c, v) for c, v, _ in _numeric_claims(answer)[0]]


# ── tolerance helpers ────────────────────────────────────────────────────────
def _tol(decimals: int) -> float:
    """Half a unit in the last written decimal place."""
    return 0.5 * 10 ** (-decimals) + 1e-9


def _rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), 1e-9)


def _direct(v: float, dec: int, e: float) -> bool:
    return abs(e - v) <= _tol(dec) or _rel(e, v) <= 0.005


def _rounded(v: float, dec: int, e: float) -> bool:
    return abs(e - v) <= 2 * _tol(dec) or _rel(e, v) <= 0.02


def _derived_ok(v: float, dec: int, e: float) -> bool:
    return abs(e - v) <= _tol(dec) or _rel(e, v) <= 0.03


def _close(a: float, b: float, rel: float = 0.02, abs_tol: float = 0.005) -> bool:
    """Kept for external callers; precision-aware paths above are used internally."""
    return a == b or abs(a - b) <= abs_tol or _rel(a, b) <= rel


def _nontrivial(v: float, dec: int) -> bool:
    """Derived arithmetic is only allowed to justify claims that aren't trivially reachable."""
    if v in (0.0, 1.0, 100.0):
        return False
    return dec > 0 or abs(v) > _SMALL_INT_MAX


# ── data classes ─────────────────────────────────────────────────────────────
@dataclass
class Check:
    claim: str
    value: float
    matched: bool
    evidence: str | None = None
    recomputed: float | None = None
    note: str | None = None


@dataclass
class Verification:
    status: str  # consistent | contradiction | unverifiable | skipped
    checks: list[Check] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── evidence ─────────────────────────────────────────────────────────────────
def region_values(region: dict[str, Any]) -> list[tuple[float, str]]:
    vals: list[tuple[float, str]] = []
    s = region.get("structured") or {}
    for c in s.get("cells", []) or []:
        if isinstance(c.get("value"), (int, float)):
            vals.append((float(c["value"]), f"cell[{c.get('row')},{c.get('col')}] \"{c.get('text','')}\""))
        else:
            for value in extract_numbers(c.get("text", "")):
                vals.append((value, f"cell[{c.get('row')},{c.get('col')}] \"{c.get('text','')}\""))
    for ser in s.get("series", []) or []:
        for p in ser.get("points", []) or []:
            if isinstance(p.get("y"), (int, float)):
                vals.append((float(p["y"]), f"{ser.get('name','series')} @ {p.get('x')}"))
    if s.get("latex"):
        for v in extract_numbers(str(s["latex"])):
            vals.append((v, f"equation R{region.get('id')}"))
    # Structured values are preferred because content often repeats the same
    # table/chart numbers and also contains row labels, years, and citation ids.
    if not vals:
        for _, v, _ in _numeric_claims(region.get("content", ""))[0]:
            vals.append((v, f"text of {region.get('type')} R{region.get('id')}"))
    return vals


def _requested_operations(claim: str) -> set[str]:
    text = claim.lower()
    operations: set[str] = set()
    if any(word in text for word in ("difference", "increase", "decrease", "higher", "lower", "more", "less", "change")):
        operations.add("difference")
    if "%" in text or any(word in text for word in ("percent", "percentage", "increase", "decrease", "change")):
        operations.add("percent")
    if any(word in text for word in ("ratio", "times", "fold", " per ", "divided")):
        operations.add("ratio")
    if any(word in text for word in ("total", "sum", "combined")):
        operations.add("sum")
    if any(word in text for word in ("average", "mean")):
        operations.add("mean")
    return operations


def derived_values(vals: list[tuple[float, str]], operations: set[str] | None = None) -> list[tuple[float, str]]:
    base = vals[:30]
    out: list[tuple[float, str]] = []
    nums = [v for v, _ in base]
    requested = operations or {"difference", "percent", "ratio", "sum", "mean"}
    if len(nums) >= 2 and "sum" in requested:
        out.append((sum(nums), "sum(cited values)"))
    if len(nums) >= 2 and "mean" in requested:
        out.append((sum(nums) / len(nums), "mean(cited values)"))
    for (a, la), (b, lb) in itertools.permutations(base, 2):
        if a == b:
            continue  # a−a=0, a÷a=1 are not evidence of anything
        if "difference" in requested:
            out.append((a - b, f"{la} − {lb}"))
        if b != 0 and "percent" in requested:
            out.append(((a - b) / abs(b) * 100, f"%Δ {lb} → {la}"))
        if b != 0 and "ratio" in requested:
            out.append((a / b, f"{la} ÷ {lb}"))
    return out


# ── main entry ───────────────────────────────────────────────────────────────
def verify(answer: str, cited_regions: list[dict[str, Any]]) -> Verification:
    claims, ignored = _numeric_claims(answer)
    note_ignored = f" ({ignored} label/citation number(s) ignored)" if ignored else ""
    if not claims:
        return Verification("skipped", [], f"No numeric claims in the answer; nothing to recompute.{note_ignored}")

    evidence_by_region = [region_values(r) for r in cited_regions]
    evidence = [value for region in evidence_by_region for value in region]
    if not evidence:
        return Verification("unverifiable",
                            [Check(c, v, False, note="no numeric evidence in cited regions") for c, v, _ in claims],
                            "Cited regions expose no numeric values to check against.")

    checks: list[Check] = []
    for claim, v, dec in claims:
        direct = next((e for e in evidence if _direct(v, dec, e[0])), None)
        if direct:
            checks.append(Check(claim, v, True, direct[1]))
            continue
        rounded = next((e for e in evidence if _rounded(v, dec, e[0])), None)
        if rounded:
            checks.append(Check(claim, v, True, f"{rounded[1]} (rounded)", note="within rounding tolerance"))
            continue
        operations = _requested_operations(claim)
        if operations and _nontrivial(v, dec):
            # Only compute operations explicitly requested by the answer, and
            # never combine unrelated values from different cited regions.
            derived = [item for region in evidence_by_region for item in derived_values(region, operations)]
            der = next((d for d in derived if _derived_ok(v, dec, d[0])), None)
            if der:
                checks.append(Check(claim, v, True, der[1], round(der[0], 4), "recomputed from cited cells"))
                continue
        nearest = min(evidence, key=lambda e: abs(e[0] - v), default=None)
        checks.append(Check(claim, v, False, nearest[1] if nearest else None,
                            round(nearest[0], 4) if nearest else None,
                            "value not found in or derivable from cited regions"))

    bad = [c for c in checks if not c.matched]
    if not bad:
        return Verification("consistent", checks,
                            f"All {len(checks)} numeric claim(s) match or recompute from the cited regions.{note_ignored}")
    return Verification("contradiction", checks,
                        f"{len(bad)}/{len(checks)} numeric claim(s) could not be reproduced from the cited regions — "
                        f"possible text-vs-figure contradiction.{note_ignored}")
