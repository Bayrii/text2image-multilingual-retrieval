"""FastAPI routes."""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from .schemas import (HealthResponse, IndexAddRequest, IndexAddResponse,
                      SearchHit, SearchRequest, SearchResponse, StatsResponse,
                      VoiceSearchResponse)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health(request: Request):
    state = request.app.state
    return HealthResponse(
        status="ok",
        index_size=state.image_indexer.size if state.image_indexer else 0,
        model=state.config["model"]["mclip_model"],
    )


@router.get("/stats", response_model=StatsResponse)
def stats(request: Request):
    state = request.app.state
    cfg = state.config
    idx_path = Path(cfg["retrieval"]["index_path"])
    size_mb = idx_path.stat().st_size / (1024 * 1024) if idx_path.exists() else 0.0
    return StatsResponse(
        total_images=state.image_indexer.size if state.image_indexer else 0,
        languages=len(cfg["data"]["languages"]),
        index_size_mb=round(size_mb, 3),
    )


@router.post("/search", response_model=SearchResponse)
def search(req: SearchRequest, request: Request):
    state = request.app.state
    if state.two_stage is None:
        raise HTTPException(500, "Retriever not initialized")

    retriever = state.two_stage if req.retriever == "two_stage" else state.text_bridge
    if retriever is None:
        raise HTTPException(503, f"Retriever '{req.retriever}' unavailable (no index loaded)")

    t0 = time.time()
    try:
        results = retriever.retrieve(req.query, top_k=req.top_k)
    except Exception as e:
        raise HTTPException(500, f"Inference failed: {e}")
    latency_ms = (time.time() - t0) * 1000

    hits = [SearchHit(
        image_id=int(r["image_id"]),
        score=float(r["score"]),
        file_path=r.get("file_path", ""),
        rank=int(r.get("rank", i + 1)),
        caption=r.get("caption"),
        language=r.get("language"),
    ) for i, r in enumerate(results)]

    request.app.state.logger.info("/search q=%r retriever=%s latency=%.1fms hits=%d",
                                  req.query, req.retriever, latency_ms, len(hits))
    return SearchResponse(query=req.query, results=hits, latency_ms=round(latency_ms, 2))


@router.post("/search/voice", response_model=VoiceSearchResponse)
async def search_voice(
    request: Request,
    audio: UploadFile = File(...),
    top_k: int = Form(5),
    language: str | None = Form(None),
    retriever: str = Form("two_stage"),
):
    state = request.app.state
    if state.voice is None:
        raise HTTPException(503, "Voice handler not loaded (set VOICE_ENABLED=1)")
    chosen = state.two_stage if retriever == "two_stage" else state.text_bridge
    if chosen is None:
        raise HTTPException(503, f"Retriever '{retriever}' unavailable")

    audio_bytes = await audio.read()
    t0 = time.time()
    try:
        info = state.voice.transcribe_bytes(audio_bytes, language=language)
    except Exception as e:
        raise HTTPException(400, f"Transcription failed: {e}")
    transcribe_ms = (time.time() - t0) * 1000

    transcript = info["text"]
    if not transcript:
        return VoiceSearchResponse(
            transcript="",
            detected_language=info.get("language", "unknown"),
            results=[],
            transcribe_ms=round(transcribe_ms, 2),
            retrieve_ms=0.0,
        )

    t1 = time.time()
    results = chosen.retrieve(transcript, top_k=top_k)
    retrieve_ms = (time.time() - t1) * 1000

    hits = [SearchHit(
        image_id=int(r["image_id"]),
        score=float(r["score"]),
        file_path=r.get("file_path", ""),
        rank=int(r.get("rank", i + 1)),
        caption=r.get("caption"),
        language=r.get("language"),
    ) for i, r in enumerate(results)]

    request.app.state.logger.info(
        "/search/voice transcript=%r lang=%s tx=%.0fms rt=%.0fms hits=%d",
        transcript, info.get("language"), transcribe_ms, retrieve_ms, len(hits))

    return VoiceSearchResponse(
        transcript=transcript,
        detected_language=info.get("language", "unknown"),
        results=hits,
        transcribe_ms=round(transcribe_ms, 2),
        retrieve_ms=round(retrieve_ms, 2),
    )


@router.post("/index/add", response_model=IndexAddResponse)
def index_add(req: IndexAddRequest, request: Request):
    state = request.app.state
    if state.model is None or state.image_indexer is None:
        raise HTTPException(500, "Model or index not initialized")

    import torch
    from PIL import Image

    new_embs = []
    new_meta = []
    with torch.no_grad():
        for item in req.images:
            if not os.path.exists(item.file_path):
                continue
            img = Image.open(item.file_path).convert("RGB")
            pixel = state.model.image_processor(images=img, return_tensors="pt")[
                "pixel_values"
            ].to(state.model.device)
            emb = state.model.encode_image(pixel)[0].float().cpu().numpy()
            new_embs.append(emb)
            new_meta.append({
                "image_id": hash(item.file_path) & 0x7FFFFFFF,
                "file_path": item.file_path,
                "caption_en": item.caption_en,
            })

    if not new_embs:
        return IndexAddResponse(added=0, index_size=state.image_indexer.size)

    state.image_indexer.add_images(np.stack(new_embs), new_meta)
    state.image_indexer.persist(state.config["retrieval"]["index_path"],
                                state.config["retrieval"]["meta_path"],
                                emb_path=state.config["retrieval"]["image_emb_path"])
    return IndexAddResponse(added=len(new_meta), index_size=state.image_indexer.size)
