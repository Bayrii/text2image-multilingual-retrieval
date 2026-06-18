"""Pydantic request/response models."""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=512)
    top_k: int = Field(5, ge=1, le=50)
    retriever: Literal["two_stage", "text_bridge"] = "two_stage"


class SearchHit(BaseModel):
    image_id: int
    score: float
    file_path: str
    image_url: str
    rank: int
    caption: Optional[str] = None
    language: Optional[str] = None


class SearchResponse(BaseModel):
    query: str
    results: List[SearchHit]
    latency_ms: float


class IndexAddItem(BaseModel):
    file_path: str
    caption_en: str


class IndexAddRequest(BaseModel):
    images: List[IndexAddItem]


class IndexAddResponse(BaseModel):
    added: int
    index_size: int


class HealthResponse(BaseModel):
    status: str
    index_size: int
    model: str


class StatsResponse(BaseModel):
    total_images: int
    languages: int
    index_size_mb: float


class VoiceSearchResponse(BaseModel):
    transcript: str
    detected_language: str
    results: List[SearchHit]
    transcribe_ms: float
    retrieve_ms: float
