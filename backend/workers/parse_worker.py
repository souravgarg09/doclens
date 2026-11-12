"""CUDA/MPS/CPU parsing worker.

Consumes {"document_id": N} jobs from Redis and runs the layout pipeline:

  PDF ─▶ render pages (PyMuPDF) ─▶ DocLayout-YOLO regions
        ├─ table    ─▶ Table Transformer  ─▶ typed cells
        ├─ figure   ─▶ DePlot             ─▶ series / axes / units (chart) or caption (figure)
        ├─ equation ─▶ pix2tex            ─▶ LaTeX (optional dependency)
        └─ text     ─▶ PyMuPDF text in bbox
       ─▶ SigLIP region embeddings ─▶ Qdrant (per-document region index) + Postgres rows

No cloud model is called while parsing: every step above runs locally.

The worker records how long each stage took (`documents.parse_stats`) and reports
live progress (`documents.progress`) so the UI and the benchmark script can show
exactly where parsing time goes.

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
import platform  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import defaultdict  # noqa: E402
from contextlib import contextmanager  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Generator  # noqa: E402

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `app` importable

import pymupdf  # noqa: E402
from PIL import Image  # noqa: E402
from sqlalchemy import delete, update  # noqa: E402

from app.config import settings  # noqa: E402
from app.db import Document, Region, SessionLocal, init_db  # noqa: E402
from app.device import DEVICE, describe as describe_device  # noqa: E402
from app.queue import blocking_pop, set_worker_status  # noqa: E402
from app.services import embeddings, vector_store  # noqa: E402
from app.services.verifier import extract_numbers  # noqa: E402
from workers.parsers import charts, equations, layout, tables  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("parse_worker")

_VISUAL_TYPES = ("table", "figure", "chart", "equation")


class StageTimer:
    """Accumulates wall-clock seconds and call counts per pipeline stage."""

    def __init__(self) -> None:
        self.seconds: dict[str, float] = defaultdict(float)
        self.counts: dict[str, int] = defaultdict(int)

    @contextmanager
    def stage(self, name: str) -> Generator[None, None, None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.seconds[name] += time.perf_counter() - start
            self.counts[name] += 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "stages": {name: round(value, 3) for name, value in self.seconds.items()},
            "counts": dict(self.counts),
        }


def _sync_device() -> None:
    """GPU kernels run asynchronously; wait for them so stage timings are honest."""
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    elif DEVICE.type == "mps":
        torch.mps.synchronize()


def preload_models() -> dict[str, float]:
    """Load every model once at start-up and return the load time of each (seconds).

    Loading (and, on a fresh machine, downloading) the weights is the slowest part of
    a first parse. Doing it before the first job arrives keeps upload-to-ready time
    stable and makes a cold start visible in the worker log instead of hiding it
    inside the first document's parse time.
    """
    loaders = {
        "layout (DocLayout-YOLO)": layout._model,
        "embeddings (SigLIP)": embeddings._load,
        "tables (Table Transformer)": tables._model,
        "charts (DePlot)": charts._model,
    }
    timings: dict[str, float] = {}
    for name, load in loaders.items():
        set_worker_status({"state": "loading", "detail": f"loading {name}"})
        start = time.perf_counter()
        load()
        timings[name] = round(time.perf_counter() - start, 2)
        log.info("model ready: %s in %.1fs", name, timings[name])
    # One warm-up page lets cuDNN pick its fastest kernels now instead of on the
    # first real document (a US-Letter page rendered at 144 dpi).
    set_worker_status({"state": "loading", "detail": "warming up the GPU"})
    start = time.perf_counter()
    with Image.new("RGB", (int(8.5 * settings.render_dpi), int(11 * settings.render_dpi)), "white") as blank:
        layout.detect_layout(blank)
        with torch.inference_mode():
            # chart decoder: the first generate() call pays ~6 s of kernel selection
            chart_proc, chart_model = charts._model()
            chart_inputs = chart_proc(images=blank.resize((600, 400)), text="Generate underlying data table of the figure below:",
                                      return_tensors="pt").to(DEVICE)
            chart_model.generate(**chart_inputs, max_new_tokens=4)
            # table structure model
            table_proc, table_model = tables._model()
            table_model(**table_proc(images=blank.resize((800, 400)), return_tensors="pt").to(DEVICE))
    _sync_device()
    timings["warm-up"] = round(time.perf_counter() - start, 2)
    log.info("warm-up done in %.1fs", timings["warm-up"])
    return timings


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


async def _set_progress(doc_id: int, stage: str, page: int, pages: int) -> None:
    """Publish progress in its own short transaction so the UI sees it immediately."""
    set_worker_status({"state": "parsing", "document_id": doc_id, "page": page, "pages": pages})
    async with SessionLocal() as s:
        await s.execute(update(Document).where(Document.id == doc_id)
                        .values(progress={"stage": stage, "page": page, "pages": pages}))
        await s.commit()


async def _process(doc_id: int) -> None:
    t0 = time.perf_counter()
    timer = StageTimer()
    async with SessionLocal() as s:
        doc = await s.get(Document, doc_id)
        if not doc:
            log.warning("document %s vanished", doc_id)
            return
        doc.status = "parsing"
        doc.error = None
        doc.progress = {"stage": "rendering", "page": 0, "pages": 0}
        doc.parse_stats = None
        await s.commit()
        pdf_path = settings.resolve_project_path(doc.storage_path)
    set_worker_status({"state": "parsing", "document_id": doc_id})

    pdf: pymupdf.Document | None = None
    try:
        # Only ever open files that the API stored in the upload directory.
        if not pdf_path.is_relative_to(settings.upload_dir):
            raise PermissionError(f"refusing to parse a file outside the upload directory: {pdf_path}")
        if not pdf_path.is_file():
            raise FileNotFoundError(f"uploaded PDF is missing: {pdf_path}")
        shutil.rmtree(settings.pages_dir / str(doc_id), ignore_errors=True)
        shutil.rmtree(settings.crops_dir / str(doc_id), ignore_errors=True)
        with timer.stage("render"):
            pdf, pages = render_pages(pdf_path, doc_id)
        n_pages = len(pages)
        with timer.stage("index_setup"):
            vector_store.ensure_collection(embeddings.dim())
            vector_store.delete_document(doc_id)
        crops_dir = settings.crops_dir / str(doc_id)
        crops_dir.mkdir(parents=True, exist_ok=True)
        rows: list[Region] = []
        points = []
        region_counts: dict[str, int] = defaultdict(int)

        async with SessionLocal() as s:
            await s.execute(delete(Region).where(Region.document_id == doc_id))
            for page_no, png, fpage in pages:
                await _set_progress(doc_id, "analysing", page_no, n_pages)
                with Image.open(png) as source:
                    img = source.convert("RGB")
                W, H = img.size
                page_regions: list[tuple[Region, Image.Image | None]] = []
                try:
                    with timer.stage("layout"):
                        detections = layout.detect_layout(img)  # {type, bbox(normalized), conf}
                        _sync_device()
                    for det in detections:
                        rtype, bbox, conf = det["type"], det["bbox"], det["conf"]
                        crop = img.crop((int(bbox[0] * W), int(bbox[1] * H), int(bbox[2] * W), int(bbox[3] * H)))
                        try:
                            if crop.width < 8 or crop.height < 8:
                                continue
                            with timer.stage("text"):
                                raw_text = text_in_bbox(fpage, bbox)
                            structured: dict = {}
                            content = raw_text
                            try:
                                if rtype == "table":
                                    with timer.stage("tables"):
                                        structured = tables.parse_table(crop, fpage, bbox)
                                        _sync_device()
                                    content = (structured.get("caption") or "") + "\n" + structured.get("markdown", raw_text)
                                elif rtype == "figure":
                                    if charts.looks_like_chart(crop, raw_text):
                                        rtype = "chart"
                                        with timer.stage("charts"):
                                            structured = charts.decode_chart(crop)
                                            _sync_device()
                                        content = structured.get("table_text", "") + "\n" + raw_text
                                    else:
                                        structured = {"caption": raw_text}
                                elif rtype == "equation":
                                    with timer.stage("equations"):
                                        latex = equations.ocr_equation(crop)
                                    structured = {"latex": latex}
                                    content = latex or raw_text
                            except Exception as e:  # noqa: BLE001 - retain the region when a specialist fails
                                log.warning("specialist parser failed on p%s %s: %s", page_no, rtype, e)
                            structured["numbers"] = extract_numbers(content)

                            crop_path = crops_dir / f"p{page_no}_{len(rows)}.png"
                            if rtype in _VISUAL_TYPES:
                                crop.save(crop_path)
                            persisted_crop = (str(crop_path.relative_to(settings.resolve_project_path(".")))
                                              if crop_path.exists() else None)
                            region = Region(document_id=doc_id, page=page_no, type=rtype,
                                            bbox=[round(v, 4) for v in bbox], content=content.strip(),
                                            structured=structured, crop_path=persisted_crop, confidence=float(conf))
                            s.add(region)
                            rows.append(region)
                            region_counts[rtype] += 1
                            page_regions.append((region, crop.copy() if rtype != "text" else None))
                        finally:
                            crop.close()

                    if page_regions:
                        await s.flush()  # assign all page region IDs in one database round trip
                        try:
                            with timer.stage("embeddings"):
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
                log.info("doc %s page %s/%s → %s regions so far (%.1fs)",
                         doc_id, page_no, n_pages, len(rows), time.perf_counter() - t0)

            if not rows:
                # Never report an empty index as "ready": questions would silently fail.
                raise RuntimeError("No regions were detected on any page. The PDF may be empty or image-only, "
                                   "or the layout model failed on this device (try TORCH_DEVICE=cpu).")
            await _set_progress(doc_id, "indexing", n_pages, n_pages)
            with timer.stage("vector_upsert"):
                for i in range(0, len(points), 128):
                    vector_store.upsert_regions(points[i:i + 128])
            total = round(time.perf_counter() - t0, 3)
            stats = {
                **timer.as_dict(),
                "total_seconds": total,
                "pages": n_pages,
                "regions": len(rows),
                "region_types": dict(region_counts),
                "seconds_per_page": round(total / max(1, n_pages), 3),
                "device": describe_device(),
                "platform": f"{platform.system()} {platform.machine()}",
                "torch": torch.__version__,
            }
            doc = await s.get(Document, doc_id)
            if not doc:
                raise RuntimeError(f"document {doc_id} was deleted while parsing")
            doc.pages = n_pages
            doc.status = "ready"
            doc.error = None
            doc.progress = None
            doc.parse_stats = stats
            await s.commit()
        log.info("doc %s READY: %s regions in %.1fs on %s · stages %s",
                 doc_id, len(rows), total, describe_device(), stats["stages"])
    except Exception as e:  # noqa: BLE001
        log.exception("doc %s FAILED", doc_id)
        async with SessionLocal() as s:
            doc = await s.get(Document, doc_id)
            if doc:
                doc.status = "failed"
                doc.error = str(e)[:2000]
                doc.progress = None
                await s.commit()
    finally:
        if pdf is not None:
            pdf.close()
        set_worker_status({"state": "idle"})


async def main_async() -> None:
    """Single event loop for the whole worker lifetime.

    asyncpg connections are bound to the loop that created them, so calling
    asyncio.run() once for init_db() and again per job ("attached to a different
    loop") crashes the worker. Everything runs on one loop; the sync Redis BRPOP
    is pushed to a thread so it does not block that loop.
    """
    await init_db()
    log.info("parse_worker starting · device=%s · queue=%s · storage=%s",
             describe_device(), settings.parse_queue, settings.data_dir)
    if settings.preload_models:
        load_times = await asyncio.to_thread(preload_models)
        log.info("all models loaded in %.1fs", sum(load_times.values()))
    set_worker_status({"state": "idle"})
    log.info("parse_worker online")
    while True:
        try:
            job = await asyncio.to_thread(blocking_pop, 5)
        except Exception:  # noqa: BLE001 — e.g. Redis hiccup; back off and retry
            log.exception("queue pop failed; retrying in 2s")
            await asyncio.sleep(2)
            continue
        if not job:
            set_worker_status({"state": "idle"})  # refresh the heartbeat
            continue
        try:
            await _process(int(job["document_id"]))
        except Exception:  # noqa: BLE001 — never let one bad job kill the worker
            log.exception("unhandled error processing job %s", job)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
