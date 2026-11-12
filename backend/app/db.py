"""Async SQLAlchemy (asyncpg) models: documents, regions, queries."""
from datetime import datetime
from typing import Any, AsyncIterator

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from .config import settings

engine = create_async_engine(settings.database_url, pool_pre_ping=True)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    title: Mapped[str] = mapped_column(String(300))
    filename: Mapped[str] = mapped_column(String(300))
    storage_path: Mapped[str] = mapped_column(Text)
    pages: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(20), default="queued")  # queued|parsing|ready|failed
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    regions: Mapped[list["Region"]] = relationship(back_populates="document", cascade="all, delete-orphan")


class Region(Base):
    """A layout region: bbox is normalized [x0,y0,x1,y1] in 0..1 page coords."""
    __tablename__ = "regions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    page: Mapped[int] = mapped_column(Integer, index=True)
    type: Mapped[str] = mapped_column(String(20))  # text|title|table|figure|chart|equation
    bbox: Mapped[list[float]] = mapped_column(JSON)
    content: Mapped[str] = mapped_column(Text, default="")
    structured: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)  # cells / series / latex
    crop_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    document: Mapped[Document] = relationship(back_populates="regions")

    def as_dict(self, *, include_paths: bool = False) -> dict[str, Any]:
        result = {"id": self.id, "document_id": self.document_id, "page": self.page, "type": self.type,
                  "bbox": self.bbox, "content": self.content, "structured": self.structured,
                  "confidence": self.confidence}
        if include_paths:
            result["crop_path"] = self.crop_path
        return result


class Query(Base):
    __tablename__ = "queries"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(20))  # answered|refused|flagged
    grounding: Mapped[float] = mapped_column(Float, default=0)
    citations: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    verification: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    trace: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


async def init_db() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_session() -> AsyncIterator[AsyncSession]:
    async with SessionLocal() as session:
        yield session
