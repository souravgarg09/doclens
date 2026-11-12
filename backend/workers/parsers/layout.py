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


def detect_layout(img: Image.Image, conf: float = 0.25) -> list[dict]:
    W, H = img.size
    # fp32 on purpose: fp16 loses every box on some GPUs (GTX 16-series) and saves little here
    res = _model().predict(img, imgsz=1024, conf=conf, device=str(DEVICE), half=False, verbose=False)[0]
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
