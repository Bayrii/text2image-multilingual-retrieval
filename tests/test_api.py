"""API tests using FastAPI TestClient with stubbed model + indexer.

These avoid loading mCLIP weights -- we monkeypatch the lifespan to inject
fakes that satisfy the route contracts.
"""
from contextlib import asynccontextmanager
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes import router
from src.api.schemas import HealthResponse


class FakeIndexer:
    def __init__(self, n=3, dim=4):
        self.size = n
        self.dim = dim

    def search(self, q, top_n):
        return [
            {"image_id": i, "file_path": f"/img/{i}.jpg",
             "language": "en", "caption": f"cap{i}", "score": 1.0 - i * 0.1}
            for i in range(min(top_n, self.size))
        ]


class FakeRetriever:
    def __init__(self, idx):
        self.idx = idx

    def retrieve(self, query, top_k=5):
        results = self.idx.search(None, top_k)
        for i, r in enumerate(results, start=1):
            r["rank"] = i
        return results


class FakeVoiceHandler:
    def __init__(self, transcript="a dog in the park", language="en"):
        self.transcript = transcript
        self.language = language

    def transcribe_bytes(self, audio_bytes, language=None):
        return {"text": self.transcript,
                "language": language or self.language}


@pytest.fixture
def client():
    @asynccontextmanager
    async def fake_lifespan(app):
        idx = FakeIndexer()
        app.state.config = {
            "model": {"mclip_model": "fake-model"},
            "data": {"languages": ["en", "fr"]},
            "retrieval": {"index_path": "/nonexistent.index"},
        }
        import logging
        app.state.logger = logging.getLogger("test")
        app.state.model = SimpleNamespace(image_processor=None,
                                          encode_image=lambda x: x, device="cpu")
        app.state.image_indexer = idx
        app.state.text_indexer = idx
        app.state.two_stage = FakeRetriever(idx)
        app.state.text_bridge = FakeRetriever(idx)
        app.state.voice = FakeVoiceHandler()
        yield

    app = FastAPI(lifespan=fake_lifespan)
    app.include_router(router)
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    HealthResponse(**data)
    assert data["status"] == "ok"


def test_search_schema(client):
    r = client.post("/search", json={"query": "hello world", "top_k": 3,
                                     "retriever": "two_stage"})
    assert r.status_code == 200
    body = r.json()
    assert body["query"] == "hello world"
    assert len(body["results"]) == 3
    assert all("rank" in h and "score" in h and "image_id" in h
               for h in body["results"])
    assert body["latency_ms"] >= 0


def test_search_validation(client):
    r = client.post("/search", json={"query": "", "top_k": 3})
    assert r.status_code == 422


def test_stats(client):
    r = client.get("/stats")
    assert r.status_code == 200
    body = r.json()
    assert "total_images" in body and "languages" in body


def test_search_voice_schema(client):
    fake_wav = b"RIFF\x00\x00\x00\x00WAVEfmt "  # not a real WAV; handler is stubbed
    r = client.post(
        "/search/voice",
        files={"audio": ("query.wav", fake_wav, "audio/wav")},
        data={"top_k": "3", "retriever": "two_stage"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["transcript"] == "a dog in the park"
    assert body["detected_language"] == "en"
    assert len(body["results"]) == 3
    assert all("rank" in h and "score" in h for h in body["results"])
    assert body["transcribe_ms"] >= 0
    assert body["retrieve_ms"] >= 0


def test_search_voice_language_hint(client):
    r = client.post(
        "/search/voice",
        files={"audio": ("q.wav", b"\x00\x00", "audio/wav")},
        data={"top_k": "1", "retriever": "two_stage", "language": "ar"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["detected_language"] == "ar"
