"""Layout detection with DocLayout-YOLO (YOLOv10-based, DocStructBench weights)."""
from functools import lru_cache

from huggingface_hub import hf_hub_download
from PIL import Image

from app.config import settings
from app.device import DEVICE

# DocStructBench class names → our region types
CLASS_MAP = {
    "title": "title", "plain text": "text", "abandon": None, "figure": "figure", "figure_caption": "text",
    "table": "table", "table_caption": "text", "table_footnote": "text", "isolate_formula": "equation", "formula_caption": "text",
}


@lru_cache(maxsize=1)
def _model():
    from doclayout_yolo import YOLOv10  # relies on the torch.load monkeypatch in parse_worker.py
    weights = hf_hub_download(repo_id=settings.layout_repo, filename=settings.layout_weights)
    return YOLOv10(weights)


_TEXTUAL = ("text", "title")


def _area(b: list[float]) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _intersection(a: list[float], b: list[float]) -> float:
    return _area([max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])])


def _norm(text: str) -> str:
    return " ".join(text.split()).lower()


def dedupe_regions(regions: list[dict]) -> list[dict]:
    """Remove empty and duplicate text/title regions; never touches tables, figures or equations.

    The layout model sometimes returns the same paragraph several times (an author block,
    then each of its lines; a footer as both text and title). Duplicates waste the few
    context slots sent to the model, so a text/title region is dropped when
      * it has no text at all, or
      * at least 90 % of its box lies inside another text/title box whose text contains
        its text (for identical boxes the lower-confidence one goes).
    Each region dict needs "type", "bbox", "conf" and "text".
    """
    keep = []
    for i, r in enumerate(regions):
        if r["type"] not in _TEXTUAL:
            keep.append(r)
            continue
        text = _norm(r.get("text", ""))
        if not text:
            continue
        duplicate = False
        for j, other in enumerate(regions):
            if i == j or other["type"] not in _TEXTUAL:
                continue
            other_text = _norm(other.get("text", ""))
            area = _area(r["bbox"])
            if not area or not other_text or text not in other_text:
                continue
            inside = _intersection(r["bbox"], other["bbox"]) / area
            if inside < 0.9:
                continue
            same = abs(area - _area(other["bbox"])) / max(area, 1e-9) < 0.1 and text == other_text
            # identical twin: keep the more confident one (ties: the earlier one)
            if same and (other["conf"] > r["conf"] or (other["conf"] == r["conf"] and j < i)):
                duplicate = True
            elif not same and _area(other["bbox"]) > area:
                duplicate = True  # contained in a larger block that already has this text
            if duplicate:
                break
        if not duplicate:
            keep.append(r)
    return keep


def detect_layout(img: Image.Image, conf: float = 0.25) -> list[dict]:
    W, H = img.size
    # fp32 on purpose: fp16 loses every box on some GPUs (GTX 16-series) and saves little here
    res = _model().predict(img, imgsz=settings.layout_imgsz, conf=conf, device=str(DEVICE), half=False, verbose=False)[0]
    names = res.names
    out = []
    for box, cls, score in zip(res.boxes.xyxy.tolist(), res.boxes.cls.tolist(), res.boxes.conf.tolist()):
        rtype = CLASS_MAP.get(names[int(cls)], "text")
        if rtype is None:
            continue
        x0, y0, x1, y1 = box
        out.append({"type": rtype, "conf": float(score),
                    "bbox": [max(0, x0 / W), max(0, y0 / H), min(1, x1 / W), min(1, y1 / H)]})
    # reading order: top→bottom, left→right
    out.sort(key=lambda r: (round(r["bbox"][1], 2), r["bbox"][0]))
    return out
