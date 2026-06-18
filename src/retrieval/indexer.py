"""FAISS indexer (flat inner-product, async-safe add).

Keeps the float32 source embeddings in memory alongside the FAISS index
so the two-stage retriever can perform an exact re-rank over a
stage-1 candidate set (matters when stage-1 is approximate, e.g. IVF).
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..utils import ensure_dir, get_logger, load_json, save_json

logger = get_logger("indexer")


def _l2norm_np(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


class FAISSIndexer:
    """Flat / IVF inner-product (cosine) FAISS index with metadata + embedding sidecars."""

    def __init__(self, dim: int = 512, index_type: str = "flat_ip"):
        self.dim = dim
        self.index_type = index_type
        self.index = None
        self.meta: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        self._lock = threading.Lock()

    @property
    def size(self) -> int:
        return 0 if self.index is None else int(self.index.ntotal)

    def _new_index(self):
        import faiss
        if self.index_type == "flat_ip":
            return faiss.IndexFlatIP(self.dim)
        if self.index_type == "ivf_flat":
            quantizer = faiss.IndexFlatIP(self.dim)
            return faiss.IndexIVFFlat(quantizer, self.dim, 100, faiss.METRIC_INNER_PRODUCT)
        raise ValueError(f"Unknown index type: {self.index_type}")

    def build(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]]) -> None:
        assert embeddings.shape[0] == len(metadata), \
            f"embeddings ({embeddings.shape[0]}) and metadata ({len(metadata)}) length mismatch"
        embeddings = _l2norm_np(embeddings.astype("float32"))
        with self._lock:
            self.index = self._new_index()
            if hasattr(self.index, "is_trained") and not self.index.is_trained:
                self.index.train(embeddings)
            self.index.add(embeddings)
            self.meta = list(metadata)
            self.embeddings = embeddings.copy()
        logger.info("Built index: ntotal=%d dim=%d", self.size, self.dim)

    def save(self, index_path: str | Path, meta_path: str | Path,
             emb_path: str | Path | None = None) -> None:
        import faiss
        index_path = Path(index_path)
        meta_path = Path(meta_path)
        ensure_dir(index_path.parent)
        with self._lock:
            faiss.write_index(self.index, str(index_path))
            save_json(self.meta, meta_path)
            if emb_path is None:
                emb_path = index_path.with_suffix(".embeddings.npy")
            else:
                emb_path = Path(emb_path)
            if self.embeddings is not None:
                np.save(emb_path, self.embeddings)
        logger.info("Saved index -> %s, meta -> %s, embeddings -> %s",
                    index_path, meta_path, emb_path)

    def load(self, index_path: str | Path, meta_path: str | Path,
             emb_path: str | Path | None = None) -> None:
        import faiss
        index_path = Path(index_path)
        with self._lock:
            self.index = faiss.read_index(str(index_path))
            self.meta = load_json(meta_path)
            self.dim = self.index.d
            if emb_path is None:
                candidate = index_path.with_suffix(".embeddings.npy")
            else:
                candidate = Path(emb_path)
            if candidate.exists():
                self.embeddings = np.load(candidate)
                if self.embeddings.shape[0] != self.size:
                    logger.warning("Embeddings rows (%d) != index size (%d); ignoring sidecar",
                                   self.embeddings.shape[0], self.size)
                    self.embeddings = None
            else:
                self.embeddings = None
                logger.warning("No embedding sidecar at %s; stage-2 re-rank will fall back to FAISS scores",
                               candidate)
        logger.info("Loaded index ntotal=%d dim=%d (embeddings cached=%s)",
                    self.size, self.dim, self.embeddings is not None)

    def search(self, query_vec: np.ndarray, top_n: int) -> List[Dict[str, Any]]:
        """Return top-N stage-1 candidates: meta dicts annotated with score and faiss row idx."""
        if self.index is None or self.size == 0:
            return []
        q = _l2norm_np(query_vec.astype("float32").reshape(1, -1))
        with self._lock:
            scores, idxs = self.index.search(q, min(top_n, self.size))
        out = []
        for score, idx in zip(scores[0].tolist(), idxs[0].tolist()):
            if idx < 0 or idx >= len(self.meta):
                continue
            entry = dict(self.meta[idx])
            entry["score"] = float(score)
            entry["_idx"] = int(idx)
            out.append(entry)
        return out

    def get_embeddings(self, indices: List[int]) -> Optional[np.ndarray]:
        if self.embeddings is None or not indices:
            return None
        return self.embeddings[np.asarray(indices, dtype=np.int64)]

    def add_images(self, new_embeddings: np.ndarray,
                   new_meta: List[Dict[str, Any]]) -> None:
        """Append new embeddings to live index (thread-safe). Index keeps growing without rebuild."""
        assert new_embeddings.shape[0] == len(new_meta)
        new_embeddings = _l2norm_np(new_embeddings.astype("float32"))
        with self._lock:
            if self.index is None:
                self.index = self._new_index()
            if hasattr(self.index, "is_trained") and not self.index.is_trained:
                self.index.train(new_embeddings)
            self.index.add(new_embeddings)
            self.meta.extend(new_meta)
            if self.embeddings is None:
                self.embeddings = new_embeddings.copy()
            else:
                self.embeddings = np.concatenate([self.embeddings, new_embeddings], axis=0)
        logger.info("Appended %d entries (ntotal=%d)", len(new_meta), self.size)

    def persist(self, index_path: str | Path, meta_path: str | Path,
                emb_path: str | Path | None = None) -> None:
        self.save(index_path, meta_path, emb_path)
