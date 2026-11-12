"""Chart decoding with Google DePlot (Pix2Struct): image → underlying data table → series."""
import re
from functools import lru_cache

import torch
from PIL import Image
from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor

from app.config import settings
from app.device import DEVICE
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


@torch.inference_mode()
def decode_chart(crop: Image.Image) -> dict:
    proc, model = _model()
    inputs = proc(images=crop.convert("RGB"), text="Generate underlying data table of the figure below:", return_tensors="pt").to(DEVICE)
    out = model.generate(**inputs, max_new_tokens=512)
    table_text = proc.decode(out[0], skip_special_tokens=True)
    return {"table_text": table_text, **_to_series(table_text)}


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
