"""Equation OCR with pix2tex (LaTeX-OCR)."""
from functools import lru_cache

from PIL import Image


@lru_cache(maxsize=1)
def _model():
    from pix2tex.cli import LatexOCR
    return LatexOCR()


def ocr_equation(crop: Image.Image) -> str:
    try:
        return _model()(crop.convert("RGB")).strip()
    except Exception:  # noqa: BLE001 — fall back to empty; caller keeps PDF text layer
        return ""
