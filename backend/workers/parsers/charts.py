"""Chart decoding with Google DePlot (Pix2Struct): image → underlying data table → series."""
import re
from functools import lru_cache

import torch
from PIL import Image
from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor, StoppingCriteria, StoppingCriteriaList

from app.config import settings
from app.device import DEVICE, autocast
from app.services.verifier import extract_numbers


@lru_cache(maxsize=1)
def _model():
    proc = Pix2StructProcessor.from_pretrained(settings.deplot_model)
    model = Pix2StructForConditionalGeneration.from_pretrained(settings.deplot_model).to(DEVICE).eval()
    return proc, model


def looks_like_chart(crop: Image.Image, caption_text: str) -> bool:
    """Cheap router: captions/axes words or high colour-uniformity typical of plots."""
    t = caption_text.lower()
    if any(w in t for w in ("plot", "chart", "curve", "axis", "vs.", "versus", "accuracy", "loss", "%", "log scale")):
        return True
    small = crop.convert("RGB").resize((64, 64))
    colours = len(set(small.getdata()))
    return colours < 900  # photos have thousands of distinct colours; plots don't


_MAX_NEW_TOKENS = 512
_CHECK_EVERY = 24


def is_table(table_text: str, final: bool = True) -> bool:
    """Does DePlot's output look like a data table?

    A real table has "|" between cells on (nearly) every row. On a diagram or a
    heat-map of words DePlot loops on single words ("It<0x0A>is<0x0A>in<0x0A>in ...")
    with no separators; those outputs are junk and are dropped.
    """
    rows = [r for r in table_text.replace("<0x0A>", "\n").split("\n") if r.strip()]
    if rows and rows[0].strip().lower().startswith("title"):
        rows = rows[1:]
    if not rows:
        return not final
    with_cells = sum(1 for r in rows if "|" in r)
    if len(rows) < 4:
        return not final or with_cells >= 1
    if with_cells / len(rows) < 0.6:
        return False
    # the other failure mode: one row (or one x label) repeated over and over,
    # e.g. "3x3, 64 | 64-d" x30 or "18-layer | 31.06 | ..." on every row
    data = [r.strip() for r in rows[1:] if "|" in r]
    if len(data) >= 8:
        if len(set(data)) / len(data) < 0.5:
            return False
        if len({r.split("|", 1)[0].strip() for r in data}) / len(data) < 0.5:
            return False
    return True


class _StopOnJunk(StoppingCriteria):
    """Stop generating as soon as the output is clearly not a table (saves up to ~40 s per figure)."""

    def __init__(self, proc: Pix2StructProcessor) -> None:
        self.proc = proc

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.BoolTensor:
        done = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        if input_ids.shape[-1] % _CHECK_EVERY == 0 and input_ids.shape[-1] >= 2 * _CHECK_EVERY:
            for i, ids in enumerate(input_ids):
                done[i] = not is_table(self.proc.decode(ids, skip_special_tokens=True), final=False)
        return done


@torch.inference_mode()
def decode_chart(crop: Image.Image) -> dict:
    """Image -> data table -> series. `is_chart` is False when no real table came out
    (a diagram or a picture); the region is then kept as a plain figure."""
    proc, model = _model()
    inputs = proc(images=crop.convert("RGB"), text="Generate underlying data table of the figure below:", return_tensors="pt").to(DEVICE)
    with autocast():
        out = model.generate(**inputs, max_new_tokens=_MAX_NEW_TOKENS,
                             stopping_criteria=StoppingCriteriaList([_StopOnJunk(proc)]))
    table_text = proc.decode(out[0], skip_special_tokens=True)
    if not is_table(table_text):
        return {"is_chart": False, "table_text": "", "series": [], "axes": {}, "title": None}
    return {"is_chart": True, "table_text": table_text, **_to_series(table_text)}


def _to_series(table_text: str) -> dict:
    """DePlot output: 'TITLE | ... <0x0A> header | s1 | s2 <0x0A> x | v1 | v2 ...'"""
    lines = [ln.strip() for ln in table_text.replace("<0x0A>", "\n").split("\n") if ln.strip()]
    title = None
    if lines and lines[0].lower().startswith("title"):
        title = lines.pop(0).split("|", 1)[-1].strip()
    if not lines:
        return {"series": [], "axes": {}, "title": title}
    header = [h.strip() for h in lines[0].split("|")]
    series = [{"name": h, "points": []} for h in header[1:]]
    for ln in lines[1:]:
        cells = [c.strip() for c in ln.split("|")]
        x = cells[0] if cells else ""
        for i, c in enumerate(cells[1:]):
            if i < len(series):
                nums = extract_numbers(c)
                if nums:
                    series[i]["points"].append({"x": x, "y": nums[0]})
    units = re.search(r"\((%|[a-zA-Z$€£]+)\)", table_text)
    return {"series": series, "axes": {"x": header[0] if header else None, "y": ", ".join(header[1:]), "units": units.group(1) if units else None}, "title": title}
