"""DocLens API — FastAPI + Uvicorn.

Run:  uvicorn app.main:app --reload --port 8000
"""
from __future__ import annotations

import logging
import shutil
import time
from uuid import uuid4
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from .agents.graph import run_agent
from .config import PROJECT_DIR, settings
from .db import Document, Query, Region, get_session, init_db
from .device import DEVICE
from .queue import enqueue_parse

log = logging.getLogger("doclens.api")


@asynccontextmanager
async def lifespan(_: FastAPI):
    await init_db()
    yield


app = FastAPI(title="DocLens API", version="0.2.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in settings.cors_origins.split(",") if origin.strip()],
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type"],
)


class AskRequest(BaseModel):
    document_id: int = Field(gt=0)
    question: str = Field(min_length=1, max_length=2000)


def _doc(d: Document) -> dict:
    return {"id": d.id, "title": d.title, "filename": d.filename, "pages": d.pages, "status": d.status, "error": d.error,
            "created_at": d.created_at.isoformat() if d.created_at else None}


@app.get("/api/health")
async def health():
    return {"ok": True, "gemini": bool(settings.gemini_api_key), "model": settings.gemini_model,
            "fallback_model": settings.gemini_fallback_model, "device": DEVICE.type}


@app.post("/api/documents", status_code=201)
async def upload(file: UploadFile = File(...), title: str | None = Form(None), session: AsyncSession = Depends(get_session)):
    original = Path(file.filename or "document.pdf").name
    if not original.lower().endswith(".pdf"):
        raise HTTPException(400, "only PDF files are supported")
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in original)[-80:] or "document.pdf"
    dest: Path = settings.upload_dir / f"{int(time.time() * 1000)}_{uuid4().hex[:8]}_{safe}"
    limit = settings.max_upload_mb * 1024 * 1024
    written = 0
    prefix = b""
    try:
        with dest.open("wb") as fh:
            while chunk := await file.read(1024 * 1024):
                if len(prefix) < 1024:
                    prefix += chunk[:1024 - len(prefix)]
                written += len(chunk)
                if written > limit:
                    raise HTTPException(413, f"file exceeds the {settings.max_upload_mb} MB limit")
                fh.write(chunk)
        if b"%PDF-" not in prefix:
            raise HTTPException(400, "the uploaded file is not a valid PDF")
    except Exception:
        dest.unlink(missing_ok=True)
        raise

    storage_path = str(dest.relative_to(PROJECT_DIR)) if dest.is_relative_to(PROJECT_DIR) else str(dest)
    document_title = (title or "").strip() or safe.rsplit(".", 1)[0]
    doc = Document(title=document_title[:300], filename=safe, storage_path=storage_path)
    session.add(doc)
    await session.commit()
    await session.refresh(doc)
    try:
        await enqueue_parse(doc.id)  # parse_worker.py picks it up
    except Exception as exc:  # keep the upload so the user can re-parse after Redis recovers
        log.exception("failed to enqueue document %s", doc.id)
        doc.status = "failed"
        doc.error = f"Could not enqueue parsing job: {exc}"
        await session.commit()
    return {"document": _doc(doc)}


@app.get("/api/documents")
async def list_documents(session: AsyncSession = Depends(get_session)):
    rows = (await session.execute(select(Document).order_by(Document.created_at.desc()))).scalars().all()
    return {"documents": [_doc(d) for d in rows]}


@app.get("/api/documents/{doc_id}")
async def get_document(doc_id: int, session: AsyncSession = Depends(get_session)):
    doc = await session.get(Document, doc_id)
    if not doc:
        raise HTTPException(404, "not found")
    regions = (await session.execute(select(Region).where(Region.document_id == doc_id).order_by(Region.page, Region.id))).scalars().all()
    return {"document": _doc(doc), "regions": [r.as_dict() for r in regions]}


@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: int, session: AsyncSession = Depends(get_session)):
    from .services import vector_store
    doc = await session.get(Document, doc_id)
    if not doc:
        raise HTTPException(404, "not found")
    upload_path = settings.resolve_project_path(doc.storage_path)
    await session.execute(delete(Document).where(Document.id == doc_id))
    await session.commit()
    try:
        vector_store.delete_document(doc_id)
    except Exception:  # noqa: BLE001 - database deletion must still complete
        log.exception("failed to delete vectors for document %s", doc_id)
    if upload_path.is_relative_to(settings.upload_dir):
        upload_path.unlink(missing_ok=True)
    shutil.rmtree(settings.pages_dir / str(doc_id), ignore_errors=True)
    shutil.rmtree(settings.crops_dir / str(doc_id), ignore_errors=True)
    return {"ok": True}


@app.post("/api/documents/{doc_id}/reparse")
async def reparse(doc_id: int, session: AsyncSession = Depends(get_session)):
    doc = await session.get(Document, doc_id)
    if not doc:
        raise HTTPException(404, "not found")
    if doc.status in ("queued", "parsing"):
        raise HTTPException(409, f"document is already {doc.status}")
    try:
        await enqueue_parse(doc_id)
    except Exception as exc:
        raise HTTPException(503, f"could not reach the parse queue: {exc}") from exc
    doc.status = "queued"
    doc.error = None
    await session.commit()
    return {"ok": True}


@app.get("/api/documents/{doc_id}/pages/{page}.png")
async def page_image(doc_id: int, page: int):
    if page < 1:
        raise HTTPException(400, "page must be at least 1")
    p = settings.pages_dir / str(doc_id) / f"{page}.png"
    if not p.exists():
        raise HTTPException(404, "page not rendered yet")
    return FileResponse(p, media_type="image/png")


@app.get("/api/regions/{region_id}/crop.png")
async def region_crop(region_id: int, session: AsyncSession = Depends(get_session)):
    r = await session.get(Region, region_id)
    crop_path = settings.resolve_project_path(r.crop_path) if r and r.crop_path else None
    if not crop_path or not crop_path.is_relative_to(settings.crops_dir) or not crop_path.is_file():
        raise HTTPException(404, "crop not found")
    return FileResponse(crop_path, media_type="image/png")


@app.get("/api/documents/{doc_id}/queries")
async def list_queries(doc_id: int, session: AsyncSession = Depends(get_session)):
    if not await session.get(Document, doc_id):
        raise HTTPException(404, "not found")
    rows = (await session.execute(select(Query).where(Query.document_id == doc_id).order_by(Query.created_at.desc()).limit(50))).scalars().all()
    return {"queries": [{"id": q.id, "question": q.question, "answer": q.answer, "status": q.status, "grounding": q.grounding,
                         "citations": q.citations, "verification": q.verification, "trace": q.trace} for q in rows]}


@app.delete("/api/queries/{query_id}")
async def delete_query(query_id: int, session: AsyncSession = Depends(get_session)):
    query = await session.get(Query, query_id)
    if not query:
        raise HTTPException(404, "query not found")
    await session.delete(query)
    await session.commit()
    return {"ok": True}


@app.delete("/api/documents/{doc_id}/queries")
async def clear_queries(doc_id: int, session: AsyncSession = Depends(get_session)):
    if not await session.get(Document, doc_id):
        raise HTTPException(404, "document not found")
    await session.execute(delete(Query).where(Query.document_id == doc_id))
    await session.commit()
    return {"ok": True}


@app.post("/api/ask")
async def ask(req: AskRequest, session: AsyncSession = Depends(get_session)):
    if not req.question.strip():
        raise HTTPException(400, "question is required")
    if not settings.gemini_api_key:
        raise HTTPException(503, "GEMINI_API_KEY is not configured")
    doc = await session.get(Document, req.document_id)
    if not doc:
        raise HTTPException(404, "document not found")
    if doc.status != "ready":
        raise HTTPException(409, f"document is not ready (status={doc.status})")
    return {"result": await run_agent(req.document_id, req.question.strip())}
