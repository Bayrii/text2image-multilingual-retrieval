"""Retriever tests using a fake encoder + small in-memory FAISS index."""
import numpy as np
import torch

from src.retrieval.indexer import FAISSIndexer
from src.retrieval.retriever import TextBridgeRetriever, TwoStageRetriever


class FakeModel:
    """Encodes text by hashing tokens into a fixed-dim vector."""

    def __init__(self, dim=8):
        self.dim = dim

    def eval(self):
        return self

    def encode_text(self, texts):
        out = []
        for t in texts:
            v = np.zeros(self.dim, dtype=np.float32)
            for i, ch in enumerate(t.lower()):
                v[i % self.dim] += (ord(ch) % 7)
            n = np.linalg.norm(v) + 1e-9
            out.append(v / n)
        return torch.tensor(np.stack(out))


CFG = {"retrieval": {"top_n_stage1": 10, "top_k_stage2": 3}}


def _build_image_index(model, captions_for_images):
    embs = model.encode_text(captions_for_images).numpy()
    meta = [{"image_id": i, "file_path": f"/img/{i}.jpg"}
            for i in range(len(captions_for_images))]
    idx = FAISSIndexer(dim=model.dim)
    idx.build(embs, meta)
    return idx


def test_two_stage_returns_top_k_sorted():
    model = FakeModel(dim=8)
    idx = _build_image_index(model, [
        "a cat on a mat",
        "a dog in the park",
        "an apple on a table",
        "a bicycle on the road",
        "a person reading a book",
    ])
    retriever = TwoStageRetriever(model, idx, CFG)
    res = retriever.retrieve("a cat on a mat", top_k=3)
    assert len(res) == 3
    scores = [r["score"] for r in res]
    assert scores == sorted(scores, reverse=True)
    assert res[0]["image_id"] == 0


def test_two_stage_uses_stored_embeddings_for_rerank():
    """After build(), embeddings are cached and the indexer can return them."""
    model = FakeModel(dim=8)
    captions = [f"cap-{i}" for i in range(7)]
    idx = _build_image_index(model, captions)
    assert idx.embeddings is not None
    block = idx.get_embeddings([0, 2, 4])
    assert block is not None
    assert block.shape == (3, 8)

    retriever = TwoStageRetriever(model, idx, CFG)
    res = retriever.retrieve("cap-3", top_k=3)
    assert len(res) == 3
    scores = [r["score"] for r in res]
    assert scores == sorted(scores, reverse=True)
    for r in res:
        assert "_idx" not in r


def test_text_bridge_groups_by_image_id():
    model = FakeModel(dim=8)
    text_meta = [
        {"image_id": 0, "file_path": "/i/0.jpg", "language": "en", "caption": "a cat"},
        {"image_id": 0, "file_path": "/i/0.jpg", "language": "fr", "caption": "un chat"},
        {"image_id": 1, "file_path": "/i/1.jpg", "language": "en", "caption": "a dog"},
        {"image_id": 1, "file_path": "/i/1.jpg", "language": "fr", "caption": "un chien"},
    ]
    embs = model.encode_text([m["caption"] for m in text_meta]).numpy()
    idx = FAISSIndexer(dim=model.dim)
    idx.build(embs, text_meta)
    retriever = TextBridgeRetriever(model, idx, CFG)
    res = retriever.retrieve("a cat", top_k=2)
    assert len(res) <= 2
    image_ids = [r["image_id"] for r in res]
    assert len(set(image_ids)) == len(image_ids), "duplicate image_ids in result"
