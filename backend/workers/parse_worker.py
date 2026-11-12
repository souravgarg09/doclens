"""DocLens CUDA/MPS/CPU parsing worker.

Consumes {"document_id": N} jobs from Redis and runs the layout pipeline:

  PDF ─▶ render pages (PyMuPDF) ─▶ DocLayout-YOLO regions
        ├─ table    ─▶ Table Transformer  ─▶ typed cells
        ├─ figure   ─▶ DePlot             ─▶ series / axes / units (chart) or caption (figure)
        ├─ equation ─▶ pix2tex            ─▶ LaTeX
        └─ text     ─▶ PyMuPDF text in bbox
       ─▶ SigLIP region embeddings ─▶ Qdrant (per-document region index) + Postgres rows

Run:  python -m workers.parse_worker
"""
from __future__ import annotations

# ── PyTorch ≥ 2.6 defaults torch.load(weights_only=True) which rejects the pickled
#    YOLOv10 / DocLayout-YOLO checkpoints. Monkeypatch BEFORE any model import. ──
import torch  # noqa: E402

_orig_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):  # type: ignore[no-untyped-def]
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)


torch.load = _patched_torch_load  # type: ignore[assignment]

import asyncio  # noqa: E402
import logging  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `app` importable

import pymupdf  # noqa: E402
from PIL import Image  # noqa: E402
from sqlalchemy import delete  # noqa: E402

from app.config import settings  # noqa: E402
from app.db import Document, Region, SessionLocal, init_db  # noqa: E402
from app.device import DEVICE  # noqa: E402
from app.queue import blocking_pop  # noqa: E402
from app.services import embeddings, vector_store  # noqa: E402
from app.services.verifier import extract_numbers  # noqa: E402
from workers.parsers.layout import detect_layout  # noqa: E402
from workers.parsers.tables import parse_table  # noqa: E402
from workers.parsers.charts import decode_chart, looks_like_chart  # noqa: E402
from workers.parsers.equations import ocr_equation  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("parse_worker")


def render_pages(pdf_path: Path, doc_id: int) -> tuple[pymupdf.Document, list[tuple[int, Path, pymupdf.Page]]]:
    out_dir = settings.pages_dir / str(doc_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = pymupdf.open(pdf_path)
    pages = []
    zoom = settings.render_dpi / 72
    for i, page in enumerate(pdf, start=1):
        png = out_dir / f"{i}.png"
        page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False).save(png)
        pages.append((i, png, page))
    return pdf, pages


def text_in_bbox(page: pymupdf.Page, bbox: list[float]) -> str:
    r = page.rect
    clip = pymupdf.Rect(bbox[0] * r.width, bbox[1] * r.height, bbox[2] * r.width, bbox[3] * r.height)
    return page.get_text("text", clip=clip).strip()


async def _process(doc_id: int) -> None:
    t0 = time.time()
    async with SessionLocal() as s:
        doc = await s.get(Document, doc_id)
        if not doc:
            log.warning("document %s vanished", doc_id); return
        doc.status = "parsing"; await s.commit()
        pdf_path = settings.resolve_project_path(doc.storage_path)

    pdf: pymupdf.Document | None = None
    try:
        if not pdf_path.is_file():
            raise FileNotFoundError(f"uploaded PDF is missing: {pdf_path}")
        shutil.rmtree(settings.pages_dir / str(doc_id), ignore_errors=True)
        shutil.rmtree(settings.crops_dir / str(doc_id), ignore_errors=True)
        pdf, pages = render_pages(pdf_path, doc_id)
        vector_store.ensure_collection(embeddings.dim())
        vector_store.delete_document(doc_id)
        crops_dir = settings.crops_dir / str(doc_id); crops_dir.mkdir(parents=True, exist_ok=True)
        rows: list[Region] = []
        points = []

        async with SessionLocal() as s:
            await s.execute(delete(Region).where(Region.document_id == doc_id))
            for page_no, png, fpage in pages:
                with Image.open(png) as source:
                    img = source.convert("RGB")
                W, H = img.size
                page_regions: list[tuple[Region, Image.Image | None]] = []
                try:
                    for det in detect_layout(img):  # {type, bbox(normalized), conf}
                        rtype, bbox, conf = det["type"], det["bbox"], det["conf"]
                        crop = img.crop((int(bbox[0] * W), int(bbox[1] * H), int(bbox[2] * W), int(bbox[3] * H)))
                        try:
                            if crop.width < 8 or crop.height < 8:
                                continue
                            raw_text = text_in_bbox(fpage, bbox)
                            structured: dict = {}
                            content = raw_text
                            try:
                                if rtype == "table":
                                    structured = parse_table(crop, fpage, bbox)
                                    content = (structured.get("caption") or "") + "\n" + structured.get("markdown", raw_text)
                                elif rtype == "figure":
                                    if looks_like_chart(crop, raw_text):
                                        rtype = "chart"
                                        structured = decode_chart(crop)
                                        content = structured.get("table_text", "") + "\n" + raw_text
                                    else:
                                        structured = {"caption": raw_text}
                                elif rtype == "equation":
                                    latex = ocr_equation(crop)
                                    structured = {"latex": latex}
                                    content = latex or raw_text
                            except Exception as e:  # noqa: BLE001 - retain the region when a specialist fails
                                log.warning("specialist parser failed on p%s %s: %s", page_no, rtype, e)
                            structured["numbers"] = extract_numbers(content)

                            crop_path = crops_dir / f"p{page_no}_{len(rows)}.png"
                            if rtype in ("table", "figure", "chart", "equation"):
                                crop.save(crop_path)
                            persisted_crop = (str(crop_path.relative_to(settings.resolve_project_path(".")))
                                              if crop_path.exists() else None)
                            region = Region(document_id=doc_id, page=page_no, type=rtype,
                                            bbox=[round(v, 4) for v in bbox], content=content.strip(),
                                            structured=structured, crop_path=persisted_crop, confidence=float(conf))
                            s.add(region)
                            rows.append(region)
                            page_regions.append((region, crop.copy() if rtype != "text" else None))
                        finally:
                            crop.close()

                    if page_regions:
                        await s.flush()  # assign all page region IDs in one database round trip
                        try:
                            vectors = embeddings.embed_regions(
                                [(crop, f"{region.type}: {region.content[:600]}")
                                 for region, crop in page_regions],
                                batch_size=settings.embedding_batch_size,
                            )
                            points.extend(
                                (region.id, vector,
                                 {"document_id": doc_id, "page": page_no, "type": region.type})
                                for (region, _), vector in zip(page_regions, vectors)
                            )
                        finally:
                            for _, embedding_crop in page_regions:
                                if embedding_crop is not None:
                                    embedding_crop.close()
                finally:
                    img.close()
                log.info("doc %s page %s/%s → %s regions so far (%.1fs)", doc_id, page_no, len(pages), len(rows), time.time() - t0)
            if points:
                for i in range(0, len(points), 128):
                    vector_store.upsert_regions(points[i:i + 128])
            doc = await s.get(Document, doc_id)
            if not doc:
                raise RuntimeError(f"document {doc_id} was deleted while parsing")
            doc.pages = len(pages); doc.status = "ready"; doc.error = None
            await s.commit()
        log.info("doc %s READY: %s regions in %.1fs on %s", doc_id, len(rows), time.time() - t0, DEVICE)
    except Exception as e:  # noqa: BLE001
        log.exception("doc %s FAILED", doc_id)
        async with SessionLocal() as s:
            doc = await s.get(Document, doc_id)
            if doc:
                doc.status = "failed"; doc.error = str(e)[:2000]; await s.commit()
    finally:
        if pdf is not None:
            pdf.close()


async def main_async() -> None:
    """Single event loop for the whole worker lifetime.

    asyncpg connections are bound to the loop that created them, so calling
    asyncio.run() once for init_db() and again per job ("attached to a different
    loop") is exactly what crashed the worker. Everything now runs on one loop;
    the sync Redis BLPOP is pushed to a thread so it doesn't block that loop.
    """
    await init_db()
    log.info("parse_worker online · device=%s · queue=%s · storage=%s", DEVICE, settings.parse_queue, settings.data_dir)
    while True:
        try:
            job = await asyncio.to_thread(blocking_pop, 5)
        except Exception:  # noqa: BLE001 — e.g. Redis hiccup; back off and retry
            log.exception("queue pop failed; retrying in 2s")
            await asyncio.sleep(2)
            continue
        if not job:
            continue
        try:
            await _process(int(job["document_id"]))
        except Exception:  # noqa: BLE001 — never let one bad job kill the worker
            log.exception("unhandled error processing job %s", job)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
