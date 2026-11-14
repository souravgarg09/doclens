"""Hyperlinks embedded in the PDF, and the reference list they point to.

Born-digital papers (LaTeX hyperref, Word, most publishers) carry invisible link
annotations: a citation "[11]" jumps to entry 11 of the reference list, "Table 2"
jumps to the table, and words like "code" or a DOI open a web address. None of
that is visible in the page text, so it is read here from the PDF's link layer:

  extract_links(pdf)      every link: where it sits, its kind, and what it points to
                          (target page + the text found at the destination)
  parse_references(pdf)   the numbered reference list ("[11] ..." / "11. ..."),
                          used to answer "what is reference 11?" and as a fallback
                          for PDFs that have no link layer (scanned journals, some
                          publisher PDFs)
  citation_numbers(text)  numbered citation markers in prose: "[4, 27-29]", "(11)"

Pure PyMuPDF, no model: a 15-page paper with 300 links takes a few milliseconds.
"""
from __future__ import annotations

import re
from typing import Any

import pymupdf

# PyMuPDF link kinds
_GOTO, _URI, _NAMED = pymupdf.LINK_GOTO, pymupdf.LINK_URI, pymupdf.LINK_NAMED

# named-destination prefixes written by LaTeX hyperref (and many other producers)
_DEST_KINDS = (
    ("cite", "citation"), ("bib", "citation"), ("ref", "citation"),
    ("table", "table"), ("figure", "figure"), ("fig", "figure"),
    ("equation", "equation"), ("eq", "equation"),
    ("section", "section"), ("subsection", "section"), ("subsubsection", "section"),
    ("chapter", "section"), ("appendix", "section"), ("paragraph", "section"),
    ("hfootnote", "footnote"), ("footnote", "footnote"), ("algorithm", "figure"),
    ("page", "page"), ("item", "item"), ("theorem", "section"), ("lemma", "section"),
)

_REF_HEADING_RE = re.compile(r"^\s*(?:\d+\.?\s*)?(references|bibliography|literature cited|works cited|reference list)\s*$", re.I)
# start of a numbered reference entry: "[11] ...", "11. ...", "11 ..." (only after the heading)
_ENTRY_RE = re.compile(r"^\s*\[(\d{1,3})\]\s*|^\s*(\d{1,3})\.\s+(?=\S)")
# numbered citation markers in running text
_MARKER_RE = re.compile(r"\[(\d{1,3}(?:\s*[,;–—-]\s*\d{1,3})*)\]|\((\d{1,3}(?:\s*[,;–—-]\s*\d{1,3})*)\)")
_DOI_RE = re.compile(r"\b10\.\d{4,9}/\S+", re.I)

_MAX_ENTRY_LINES = 7
_MAX_TEXT = 480


# ── page text as lines ───────────────────────────────────────────────────────
def _lines(page: pymupdf.Page) -> list[tuple[pymupdf.Rect, str]]:
    """Text lines of a page, with pieces of one visual line joined.

    Justified text and mixed fonts are often stored as several "lines" on the same
    baseline ("[2] C. M. Bishop." + "Neural networks for ..."); pieces closer than a
    few points are merged so a reference entry reads as one line. Reading order
    (column by column) is kept.
    """
    raw: list[tuple[pymupdf.Rect, str]] = []
    for block in page.get_text("dict", flags=pymupdf.TEXTFLAGS_TEXT)["blocks"]:
        for line in block.get("lines", []):
            text = "".join(span["text"] for span in line["spans"]).strip()
            if text:
                raw.append((pymupdf.Rect(line["bbox"]), text))
    # keep the PDF's reading order (column by column); only neighbours are merged
    out: list[tuple[pymupdf.Rect, str]] = []
    for rect, text in raw:
        if out:
            prev, prev_text = out[-1]
            if abs(rect.y0 - prev.y0) < 2 and abs(rect.y1 - prev.y1) < 2 and 0 <= rect.x0 - prev.x1 < 40:
                out[-1] = (prev | rect, f"{prev_text} {text}")
                continue
        out.append((rect, text))
    return out


def _words_in(words: list[tuple], rect: pymupdf.Rect) -> str:
    """The words whose centre lies inside rect (a link's visible text).

    `words` is page.get_text("words"), read once per page: re-reading it for each of a
    paper's few hundred links made this step take seconds.
    """
    x0, y0, x1, y1 = rect.x0, rect.y0, rect.x1, rect.y1
    inside = [w for w in words if x0 <= (w[0] + w[2]) / 2 <= x1 and y0 <= (w[1] + w[3]) / 2 <= y1]
    if not inside:  # a link over part of a word: take the words it overlaps
        inside = [w for w in words if w[0] < x1 and w[2] > x0 and w[1] < y1 and w[3] > y0]
    inside.sort(key=lambda w: (round(w[3]), w[0]))
    return " ".join(w[4] for w in inside)


def _clean(text: str) -> str:
    text = re.sub(r"-\s*\n\s*(?=[a-z])", "", text)  # re-join hyphenated line breaks
    return re.sub(r"\s+", " ", text).strip()


def _entry_at(lines: list[tuple[pymupdf.Rect, str]], x: float, y: float) -> tuple[pymupdf.Rect | None, str]:
    """The text block that starts at (x, y): a reference entry, caption or heading.

    Collects the first line in the same column at or just below y, then continuation
    lines until the next entry starts (a line back at the first line's indent, a new
    "[n]" / "n." marker, or a vertical gap).
    """
    column = [(r, t) for r, t in lines if r.x0 - 24 <= x <= r.x1 + 6 and r.y1 > y - 2]  # allows indented continuation lines
    column.sort(key=lambda item: (round(item[0].y0, 1), item[0].x0))
    start = next(((r, t) for r, t in column if r.y0 >= y - 6), None)
    if start is None or start[0].y0 - y > 40:
        return None, ""
    first, text = start
    rect, parts = pymupdf.Rect(first), [text]
    height = max(6.0, first.height)
    hanging = None  # x of continuation lines, learned from the second line
    previous = first
    for r, t in column:
        if r.y0 <= first.y0 + 1 or len(parts) >= _MAX_ENTRY_LINES:
            continue
        if r.y0 - previous.y1 > 0.9 * height or _ENTRY_RE.match(t):
            break
        if hanging is None:
            if abs(r.x0 - first.x0) <= 1.5 and _ENTRY_RE.match(text):
                hanging = r.x0  # numbered list without hanging indent
            elif r.x0 > first.x0 + 2:
                hanging = r.x0  # hanging indent: continuation lines are indented
            elif abs(r.x0 - first.x0) <= 1.5:
                hanging = r.x0  # flush paragraph (caption, heading)
            else:
                break
        elif r.x0 < hanging - 2:
            break  # back at the outdent: a new entry starts
        parts.append(t)
        rect |= r
        previous = r
    return rect, _clean("\n".join(parts))[:_MAX_TEXT]


def _dest_kind(name: str) -> str:
    lowered = name.lower()
    for prefix, kind in _DEST_KINDS:
        if lowered.startswith(prefix):
            return kind
    return "internal"


def _uri_kind(uri: str) -> str:
    lowered = uri.lower()
    if lowered.startswith("mailto:"):
        return "email"
    if "doi.org/" in lowered or _DOI_RE.search(lowered):
        return "doi"
    return "url"


def _label(text: str) -> str | None:
    """The number of a citation link's own text: "[11]", "11,", "(11)" -> "11"."""
    match = re.fullmatch(r"[\[(]?\s*(\d{1,3})\s*[\]),;.]{0,2}", text.strip())
    return match.group(1) if match else None


def _norm(rect: pymupdf.Rect, page: pymupdf.Page) -> list[float]:
    w, h = page.rect.width, page.rect.height
    return [round(max(0.0, min(1.0, v)), 4) for v in (rect.x0 / w, rect.y0 / h, rect.x1 / w, rect.y1 / h)]


# ── links ────────────────────────────────────────────────────────────────────
def extract_links(pdf: pymupdf.Document) -> list[dict[str, Any]]:
    """Every link annotation in the document, resolved to its destination.

    Each item: page (1-based), bbox (normalized), text (what the reader sees), kind
    (citation|table|figure|equation|section|footnote|page|internal|url|doi|email),
    label (citation number, if any), uri, target_page (1-based), target_bbox, target_text.
    """
    lines_cache: dict[int, list[tuple[pymupdf.Rect, str]]] = {}

    def page_lines(index: int) -> list[tuple[pymupdf.Rect, str]]:
        if index not in lines_cache:
            lines_cache[index] = _lines(pdf[index])
        return lines_cache[index]

    out: list[dict[str, Any]] = []
    for page in pdf:
        seen: set[tuple[int, ...]] = set()
        page_links = page.get_links()
        words = page.get_text("words") if page_links else []
        for link in page_links:
            src = pymupdf.Rect(link["from"])
            if src.is_empty or src.width < 1:
                continue
            key = tuple(round(v) for v in src)
            if key in seen:
                continue
            seen.add(key)
            text = _clean(_words_in(words, src))[:200]
            item: dict[str, Any] = {"page": page.number + 1, "bbox": _norm(src, page), "text": text,
                                    "kind": "internal", "label": None, "uri": None,
                                    "target_page": None, "target_bbox": None, "target_text": ""}
            kind = link.get("kind")
            if kind == _URI:
                uri = (link.get("uri") or "").strip()
                if not uri or uri.lower().startswith(("javascript:", "file:")):
                    continue
                item.update(kind=_uri_kind(uri), uri=uri[:1000])
                out.append(item)
                continue
            if kind not in (_GOTO, _NAMED):
                continue  # links to other files / launch actions are ignored
            target = link.get("page", -1)
            if target is None or not 0 <= target < pdf.page_count:
                continue
            name = link.get("nameddest") or link.get("name") or ""
            item["kind"] = _dest_kind(name) if name else "internal"
            item["label"] = _label(text) if item["kind"] in ("citation", "internal") else None
            point = link.get("to")
            tpage = pdf[target]
            tlines = page_lines(target)
            rect, entry = None, ""
            if point is not None:
                # PyMuPDF reports named destinations in PDF space (origin bottom-left) and
                # direct GoTo targets in page space; try the likely one first, keep the one
                # whose text starts with the cited number (or, failing that, any text).
                ys = [tpage.rect.height - point.y, point.y] if kind == _NAMED else [point.y, tpage.rect.height - point.y]
                found = [_entry_at(tlines, point.x, y) for y in ys]
                if item["label"]:
                    numbered = [f for f in found if (m := _ENTRY_RE.match(f[1])) and item["label"] in m.groups()]
                    if not numbered:
                        # the destination sits a line above the entry: take the entry with
                        # that number closest to it in the same column
                        near = [(min(abs(r.y0 - y) for y in ys), r) for r, t in tlines
                                if (m := _ENTRY_RE.match(t)) and item["label"] in m.groups()
                                and r.x0 - 24 <= point.x <= r.x1 + 6]
                        near = [n for n in near if n[0] < 40]
                        if near:
                            r = min(near, key=lambda n: n[0])[1]
                            numbered = [_entry_at(tlines, r.x0 + 1, r.y0)]
                    if numbered:
                        found = numbered
                rect, entry = next((f for f in found if f[1]), (None, ""))
            if item["kind"] == "internal" and entry:
                # unnamed destination: infer the kind from what is there
                if _ENTRY_RE.match(entry) and item["label"]:
                    item["kind"] = "citation"
                elif re.match(r"(table|tab\.)\s*\d", entry, re.I):
                    item["kind"] = "table"
                elif re.match(r"(figure|fig\.)\s*\d", entry, re.I):
                    item["kind"] = "figure"
            item.update(target_page=target + 1, target_text=entry,
                        target_bbox=_norm(rect, tpage) if rect is not None else None)
            out.append(item)
    return out


# ── reference list ───────────────────────────────────────────────────────────
def parse_references(pdf: pymupdf.Document) -> dict[str, dict[str, Any]]:
    """The numbered reference list: {"11": {"page", "bbox", "text"}, ...}.

    Starts at the last "References"/"Bibliography" heading and accepts entries only
    in increasing order (1, 2, 3 ...), so page numbers, volume numbers and table rows
    are never mistaken for entries. Author-year bibliographies (no numbers) return {}.
    """
    start_page, start_y = None, 0.0
    for page in pdf:
        for rect, text in _lines(page):
            if _REF_HEADING_RE.match(text):
                start_page, start_y = page.number, rect.y1  # keep the last heading
    if start_page is None:
        return {}

    starts: dict[str, tuple[int, pymupdf.Rect]] = {}
    page_lines: dict[int, list[tuple[pymupdf.Rect, str]]] = {}
    expected = 1
    for index in range(start_page, pdf.page_count):
        page = pdf[index]
        h = page.rect.height
        lines = [(r, t) for r, t in _lines(page) if r.y1 > 0.055 * h and r.y0 < 0.935 * h]  # no running header/footer
        page_lines[index] = lines
        if index == start_page:
            heading_x = next((r.x0 for r, t in lines if _REF_HEADING_RE.match(t) and abs(r.y1 - start_y) < 0.5), 0.0)
            # below the heading in its column, then everything in later columns
            lines = [(r, t) for r, t in lines if r.y0 >= start_y - 1 or r.x0 > heading_x + 60]
        for rect, text in lines:
            match = _ENTRY_RE.match(text)
            number = int(next(g for g in match.groups() if g)) if match else None
            if number is not None and expected <= number <= expected + 2:
                starts[str(number)] = (index, rect)
                expected = number + 1
        if expected == 1 and index > start_page:
            break  # no numbered list after the heading
    if len(starts) < 3:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for number, (index, rect) in starts.items():
        entry_rect, text = _entry_at(page_lines[index], rect.x0 + 1, rect.y0)
        text = re.sub(r"^\[?\d{1,3}[\].]\s*", "", text)
        out[number] = {"page": index + 1, "bbox": _norm(entry_rect or rect, pdf[index]), "text": text}
    return out


def citation_numbers(text: str, known: set[str] | None = None, max_span: int = 15) -> list[str]:
    """Numbered citation markers in prose, ranges expanded: "[4, 27-29]" -> 4, 27, 28, 29.

    Only numbers present in `known` (the parsed reference list) are returned, which
    keeps equation numbers, group sizes and the like out.
    """
    found: list[str] = []
    for bracket, paren in _MARKER_RE.findall(text or ""):
        group = bracket or paren
        for part in re.split(r"\s*[,;]\s*", group):
            bounds = re.split(r"\s*[–—-]\s*", part)
            try:
                lo, hi = int(bounds[0]), int(bounds[-1])
            except ValueError:
                continue
            if hi < lo or hi - lo > max_span:
                hi = lo
            for n in range(lo, hi + 1):
                key = str(n)
                if (known is None or key in known) and key not in found:
                    found.append(key)
    return found


# ── attaching links to layout regions ────────────────────────────────────────
_STRUCTURED_FOR = {"table": ("table",), "figure": ("figure", "chart"), "equation": ("equation",)}


def _region_at(regions: list[dict[str, Any]], x: float, y: float, prefer: tuple[str, ...] = ()) -> int | None:
    """Region containing the normalized point; otherwise the nearest one starting just below it."""
    inside = [r for r in regions if r["bbox"][0] - 0.005 <= x <= r["bbox"][2] + 0.005
              and r["bbox"][1] - 0.005 <= y <= r["bbox"][3] + 0.005]
    if prefer:
        preferred = [r for r in inside if r["type"] in prefer]
        inside = preferred or inside
    if inside:
        return min(inside, key=lambda r: (r["bbox"][2] - r["bbox"][0]) * (r["bbox"][3] - r["bbox"][1]))["id"]
    below = [r for r in regions if r["bbox"][0] - 0.02 <= x <= r["bbox"][2] + 0.02 and 0 <= r["bbox"][1] - y < 0.06]
    return min(below, key=lambda r: r["bbox"][1] - y)["id"] if below else None


def _nearest_structured(regions: list[dict[str, Any]], bbox: list[float], types: tuple[str, ...]) -> int | None:
    """The table/figure a caption belongs to: same page, closest vertically, overlapping horizontally."""
    candidates = [r for r in regions if r["type"] in types
                  and min(r["bbox"][2], bbox[2]) - max(r["bbox"][0], bbox[0]) > -0.05]
    if not candidates:
        return None

    def gap(r: dict[str, Any]) -> float:
        return max(0.0, r["bbox"][1] - bbox[3], bbox[1] - r["bbox"][3])

    best = min(candidates, key=gap)
    return best["id"] if gap(best) < 0.25 else None


def build_link_rows(pdf: pymupdf.Document, regions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Link rows for the database: embedded links, reference entries and inferred citations.

    `regions` are the document's layout regions as {id, page, type, bbox, content}.
    Citation markers in the text ("(11)", "[4, 27]") become inferred citation links
    only when the PDF has no citation links of its own but does have a numbered
    reference list, so a journal PDF without hyperlinks still resolves "reference 11".
    """
    by_page: dict[int, list[dict[str, Any]]] = {}
    for r in regions:
        by_page.setdefault(r["page"], []).append(r)

    rows: list[dict[str, Any]] = []
    references = parse_references(pdf)
    reference_region: dict[str, int | None] = {}
    for number, entry in references.items():
        b = entry["bbox"]
        region_id = _region_at(by_page.get(entry["page"], []), b[0] + 0.01, (b[1] + b[3]) / 2)
        reference_region[number] = region_id
        rows.append({"kind": "reference", "page": entry["page"], "bbox": b, "text": entry["text"], "label": number,
                     "uri": None, "target_page": None, "target_bbox": None, "target_text": "",
                     "region_id": region_id, "target_region_id": None, "inferred": False})

    links = extract_links(pdf)
    for link in links:
        b = link["bbox"]
        region_id = _region_at(by_page.get(link["page"], []), (b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
        target_region_id = None
        if link["target_page"]:
            page_regions = by_page.get(link["target_page"], [])
            tb = link["target_bbox"]
            if link["kind"] in _STRUCTURED_FOR and tb:
                target_region_id = _nearest_structured(page_regions, tb, _STRUCTURED_FOR[link["kind"]])
            if target_region_id is None and tb:
                target_region_id = _region_at(page_regions, tb[0] + 0.01, tb[1] + 0.005)
            if link["kind"] == "citation" and link["label"] in references:
                # the reference list is the authority for the entry's full text
                link["target_text"] = references[link["label"]]["text"]
        rows.append({**link, "region_id": region_id, "target_region_id": target_region_id, "inferred": False})

    if references and not any(link["kind"] == "citation" for link in links):
        known = set(references)
        reference_ids = {rid for rid in reference_region.values() if rid}
        for r in regions:
            if r["type"] not in ("text", "title", "table") or r["id"] in reference_ids:
                continue
            numbers = citation_numbers(r.get("content") or "", known)
            if not numbers:
                continue
            page = pdf[r["page"] - 1]
            w, h = page.rect.width, page.rect.height
            clip = pymupdf.Rect(r["bbox"][0] * w, r["bbox"][1] * h, r["bbox"][2] * w, r["bbox"][3] * h)
            for number in numbers:
                entry = references[number]
                hit = None
                for marker in (f"({number})", f"[{number}]", f"{number},", f"{number})", f"{number}]"):
                    found = [q for q in page.search_for(marker, clip=clip)]
                    if found:
                        hit = found[0]
                        break
                bbox = _norm(hit, page) if hit is not None else r["bbox"]
                rows.append({"kind": "citation", "page": r["page"], "bbox": bbox, "text": f"({number})",
                             "label": number, "uri": None, "target_page": entry["page"],
                             "target_bbox": entry["bbox"], "target_text": entry["text"], "region_id": r["id"],
                             "target_region_id": reference_region.get(number), "inferred": True})
    return rows
