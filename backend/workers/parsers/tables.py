"""Table structure extraction with Microsoft Table Transformer (TATR).

Rows/columns are detected on the crop; cell *text* is read from the PDF text
layer within each cell rectangle (born-digital PDFs), giving typed cells with
numeric values that the verifier can recompute from.
"""
import json
from functools import lru_cache
from pathlib import Path

import pymupdf
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import AutoImageProcessor, TableTransformerConfig, TableTransformerForObjectDetection

from app.config import settings
from app.device import DEVICE, autocast
from app.services.verifier import extract_numbers


def _config() -> TableTransformerConfig:
    """Load the 2023 model config, repairing a field newer transformers validates strictly.

    The published config has "dilation": null. transformers 5 requires a bool and refuses
    to load the model; null always meant "no dilation", so it is set to False explicitly.
    """
    path = hf_hub_download(settings.table_model, "config.json")
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if raw.get("dilation") is None:
        raw["dilation"] = False
    return TableTransformerConfig.from_dict(raw)


@lru_cache(maxsize=1)
def _model():
    # The model's saved processor config only has {"longest_edge": 1000}; transformers ≥ 4.52
    # rejects that and every table failed silently. Give it the DETR-style pair explicitly
    # (shortest side 800 px, longest side capped at 1000 px, as the TATR authors resize).
    proc = AutoImageProcessor.from_pretrained(settings.table_model,
                                              size={"shortest_edge": 800, "longest_edge": 1000})
    model = TableTransformerForObjectDetection.from_pretrained(settings.table_model, config=_config())
    return proc, model.to(DEVICE).eval()


@torch.inference_mode()
def parse_table(crop: Image.Image, page: pymupdf.Page, bbox: list[float]) -> dict:
    proc, model = _model()
    inputs = proc(images=crop, return_tensors="pt").to(DEVICE)
    with autocast():
        outputs = model(**inputs)
    det = proc.post_process_object_detection(outputs, threshold=0.5, target_sizes=[crop.size[::-1]])[0]
    labels = model.config.id2label
    rows, cols = [], []
    for box, lab in zip(det["boxes"].tolist(), det["labels"].tolist()):
        name = labels[int(lab)]
        if name == "table row": rows.append(box)
        elif name == "table column": cols.append(box)
    rows.sort(key=lambda b: b[1]); cols.sort(key=lambda b: b[0])

    # map crop pixels → PDF points
    R = page.rect
    cw, ch = crop.size
    tx0, ty0 = bbox[0] * R.width, bbox[1] * R.height
    sx, sy = (bbox[2] - bbox[0]) * R.width / cw, (bbox[3] - bbox[1]) * R.height / ch

    cells, grid = [], []
    for ri, r in enumerate(rows):
        line = []
        for ci, c in enumerate(cols):
            rect = pymupdf.Rect(tx0 + c[0] * sx, ty0 + r[1] * sy, tx0 + c[2] * sx, ty0 + r[3] * sy)
            text = " ".join(page.get_text("text", clip=rect).split())
            nums = extract_numbers(text)
            cells.append({"row": ri, "col": ci, "text": text, "value": nums[0] if len(nums) == 1 else None})
            line.append(text)
        grid.append(line)
    header_row = grid[0] if grid else []
    markdown = "\n".join(" | ".join(r) for r in grid)
    return {"cells": cells, "headers": header_row, "n_rows": len(rows), "n_cols": len(cols), "markdown": markdown}
