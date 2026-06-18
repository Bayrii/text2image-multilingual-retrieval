"""FastAPI application: load model + index once at startup, serve /search."""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from ..utils import get_logger, load_config, setup_caches
from .routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_caches()
    cfg = load_config()
    log = get_logger("api")
    log.info("Loading model and indexes...")

    from ..models.mclip import MultilingualCLIPModel
    from ..retrieval.indexer import FAISSIndexer
    from ..retrieval.retriever import TextBridgeRetriever, TwoStageRetriever
    from ..utils import get_device
    import torch

    device = get_device()
    model = MultilingualCLIPModel(cfg).to(device)

    ckpt_path = Path(cfg["training"]["checkpoints_dir"]) / "best.pt"
    if ckpt_path.exists():
        log.info("Loading fine-tuned checkpoint: %s", ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"], strict=False)
    else:
        log.warning("No fine-tuned checkpoint found at %s; using pretrained weights",
                    ckpt_path)
    model.eval()

    image_indexer = FAISSIndexer(dim=cfg["model"]["embedding_dim"],
                                 index_type=cfg["retrieval"]["faiss_index_type"])
    img_idx_p = Path(cfg["retrieval"]["index_path"])
    img_meta_p = Path(cfg["retrieval"]["meta_path"])
    if img_idx_p.exists() and img_meta_p.exists():
        image_indexer.load(img_idx_p, img_meta_p,
                           emb_path=cfg["retrieval"]["image_emb_path"])
    else:
        log.warning("No image index found; /search two_stage will be unavailable")

    text_indexer = FAISSIndexer(dim=cfg["model"]["embedding_dim"],
                                index_type=cfg["retrieval"]["faiss_index_type"])
    txt_idx_p = Path(cfg["retrieval"]["text_index_path"])
    txt_meta_p = Path(cfg["retrieval"]["text_meta_path"])
    if txt_idx_p.exists() and txt_meta_p.exists():
        text_indexer.load(txt_idx_p, txt_meta_p,
                          emb_path=cfg["retrieval"]["text_emb_path"])

    two_stage = TwoStageRetriever(model, image_indexer, cfg) if image_indexer.size > 0 else None
    text_bridge = TextBridgeRetriever(model, text_indexer, cfg) if text_indexer.size > 0 else None

    voice = None
    import os as _os
    if _os.environ.get("VOICE_ENABLED", "1") not in ("0", "false", "False"):
        try:
            from ..voice import VoiceHandler
            ws = _os.environ.get("WHISPER_SIZE", "small")
            wd = _os.environ.get("WHISPER_DEVICE", "cpu")
            voice = VoiceHandler(model_size=ws, device=wd)
            log.info("Voice handler ready (whisper=%s on %s)", ws, wd)
        except Exception as e:
            log.warning("Voice handler unavailable: %s", e)

    app.state.config = cfg
    app.state.logger = log
    app.state.model = model
    app.state.image_indexer = image_indexer
    app.state.text_indexer = text_indexer
    app.state.two_stage = two_stage
    app.state.text_bridge = text_bridge
    app.state.voice = voice

    log.info("Startup complete. image_index=%d text_index=%d",
             image_indexer.size, text_indexer.size)
    yield
    log.info("Shutting down.")


def create_app() -> FastAPI:
    app = FastAPI(title="Multilingual Text-to-Image Retrieval", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        t0 = time.time()
        response = await call_next(request)
        dt = (time.time() - t0) * 1000
        logger = logging.getLogger("api")
        logger.info("%s %s -> %d %.1fms", request.method, request.url.path,
                    response.status_code, dt)
        return response

    app.include_router(router)
    return app


app = create_app()
