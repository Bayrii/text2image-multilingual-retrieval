"""Recall@K, mAP@K, and per-language aggregation."""
from __future__ import annotations

from typing import Dict, List, Tuple


def recall_at_k(query_image_ids: List[int],
                retrieved_image_ids: List[List[int]],
                k: int) -> float:
    if not query_image_ids:
        return 0.0
    hits = 0
    for gt, retrieved in zip(query_image_ids, retrieved_image_ids):
        if gt in retrieved[:k]:
            hits += 1
    return hits / len(query_image_ids)


def average_precision(gt: int, retrieved: List[int], k: int) -> float:
    """AP for a single query. Since we have one ground truth, AP@K = 1/rank if hit, else 0."""
    for i, r in enumerate(retrieved[:k]):
        if r == gt:
            return 1.0 / (i + 1)
    return 0.0


def mean_average_precision(query_image_ids: List[int],
                           retrieved_image_ids: List[List[int]],
                           k: int) -> float:
    if not query_image_ids:
        return 0.0
    return sum(average_precision(gt, retr, k)
               for gt, retr in zip(query_image_ids, retrieved_image_ids)) / len(query_image_ids)


def compute_all_metrics(
    results_by_language: Dict[str, List[Tuple[int, List[int]]]],
    recall_ks: List[int],
    map_k: int,
) -> Dict[str, Dict[str, float]]:
    """Input: {lang: [(query_image_id, [retrieved_ids]), ...]}
    Output: {lang: {R@1, R@5, R@10, mAP@K}, "overall": {...}}
    """
    out: Dict[str, Dict[str, float]] = {}
    all_gts: List[int] = []
    all_retr: List[List[int]] = []

    for lang, pairs in results_by_language.items():
        if not pairs:
            continue
        gts = [p[0] for p in pairs]
        retr = [p[1] for p in pairs]
        m = {}
        for k in recall_ks:
            m[f"R@{k}"] = recall_at_k(gts, retr, k)
        m[f"mAP@{map_k}"] = mean_average_precision(gts, retr, map_k)
        m["n_queries"] = len(gts)
        out[lang] = m
        all_gts.extend(gts)
        all_retr.extend(retr)

    overall = {}
    for k in recall_ks:
        overall[f"R@{k}"] = recall_at_k(all_gts, all_retr, k)
    overall[f"mAP@{map_k}"] = mean_average_precision(all_gts, all_retr, map_k)
    overall["n_queries"] = len(all_gts)
    out["overall"] = overall
    return out
