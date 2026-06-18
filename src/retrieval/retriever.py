"""Two-stage and text-bridge retrievers.

TwoStageRetriever:
    Stage 1: FAISS top-N over the IMAGE index (fast, possibly approximate).
    Stage 2: exact dot-product re-rank using the cached float32 embeddings,
             then keep top-K. This is a real re-rank for IVF/HNSW indexes
             and a precision-stable confirmation step for flat_ip.

TextBridgeRetriever:
    Stage 1: FAISS top-N over the TEXT (caption) index.
    Stage 2: group by image_id, keep best per image, return top-K.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import torch

from ..utils import get_logger

logger = get_logger("retriever")


def _l2norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v) + 1e-9
    return v / n


class TwoStageRetriever:
    """FAISS top-N candidates -> exact CLIP re-rank -> top-K."""

    def __init__(self, model, image_indexer, config: Dict[str, Any]):
        self.model = model
        self.image_indexer = image_indexer
        self.top_n = int(config["retrieval"]["top_n_stage1"])
        self.top_k = int(config["retrieval"]["top_k_stage2"])

    @torch.no_grad()
    def _encode_query(self, query: str) -> np.ndarray:
        if hasattr(self.model, "eval"):
            self.model.eval()
        emb = self.model.encode_text([query])
        if hasattr(emb, "detach"):
            emb = emb.detach().float().cpu().numpy()
        else:
            emb = np.asarray(emb, dtype=np.float32)
        return _l2norm(emb[0])

    @torch.no_grad()
    def retrieve(self, query: str, top_k: int | None = None) -> List[Dict[str, Any]]:
        top_k = top_k or self.top_k
        q = self._encode_query(query)

        candidates = self.image_indexer.search(q, self.top_n)
        if not candidates:
            return []

        cand_idx = [c.get("_idx", -1) for c in candidates]
        emb_block = None
        if all(i >= 0 for i in cand_idx):
            emb_block = self.image_indexer.get_embeddings(cand_idx)

        if emb_block is not None:
            rerank_scores = (emb_block @ q.reshape(-1, 1)).ravel()
            order = np.argsort(-rerank_scores)
            re_ranked = []
            for new_rank, pos in enumerate(order[:top_k], start=1):
                item = dict(candidates[int(pos)])
                item["score"] = float(rerank_scores[int(pos)])
                item["rank"] = new_rank
                item.pop("_idx", None)
                re_ranked.append(item)
            return re_ranked

        re_ranked = sorted(candidates, key=lambda x: -x["score"])[:top_k]
        for rank, item in enumerate(re_ranked, start=1):
            item["rank"] = rank
            item.pop("_idx", None)
        return re_ranked


class TextBridgeRetriever:
    """Caption-index match, then group by image_id."""

    def __init__(self, model, text_indexer, config: Dict[str, Any]):
        self.model = model
        self.text_indexer = text_indexer
        self.top_n = int(config["retrieval"]["top_n_stage1"])
        self.top_k = int(config["retrieval"]["top_k_stage2"])

    @torch.no_grad()
    def _encode_query(self, query: str) -> np.ndarray:
        if hasattr(self.model, "eval"):
            self.model.eval()
        emb = self.model.encode_text([query])
        if hasattr(emb, "detach"):
            emb = emb.detach().float().cpu().numpy()
        else:
            emb = np.asarray(emb, dtype=np.float32)
        return _l2norm(emb[0])

    @torch.no_grad()
    def retrieve(self, query: str, top_k: int | None = None) -> List[Dict[str, Any]]:
        top_k = top_k or self.top_k
        q = self._encode_query(query)
        candidates = self.text_indexer.search(q, self.top_n)
        best_per_image: Dict[int, Dict[str, Any]] = {}
        for c in candidates:
            iid = int(c["image_id"])
            if iid not in best_per_image or c["score"] > best_per_image[iid]["score"]:
                best_per_image[iid] = c
        ranked = sorted(best_per_image.values(), key=lambda x: -x["score"])[:top_k]
        for rank, item in enumerate(ranked, start=1):
            item["rank"] = rank
            item.pop("_idx", None)
        return ranked
