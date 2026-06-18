"""Run evaluation on the test split, grouped by language."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Tuple

from tqdm import tqdm

from ..utils import get_logger
from .metrics import compute_all_metrics

logger = get_logger("evaluator")


class Evaluator:
    def __init__(self, retriever, test_rows: List[Dict[str, Any]],
                 config: Dict[str, Any]):
        self.retriever = retriever
        self.test_rows = test_rows
        self.recall_ks = list(config["evaluation"]["recall_at_k"])
        self.map_k = int(config["evaluation"]["map_at_k"])
        self.top_k_eval = max(self.recall_ks + [self.map_k])

    def run(self) -> Dict[str, Any]:
        results_by_lang: Dict[str, List[Tuple[int, List[int]]]] = defaultdict(list)
        for row in tqdm(self.test_rows, desc="eval"):
            query = row["caption"]
            lang = row["language"]
            gt_id = int(row["image_id"])
            results = self.retriever.retrieve(query, top_k=self.top_k_eval)
            retrieved_ids = [int(r["image_id"]) for r in results]
            results_by_lang[lang].append((gt_id, retrieved_ids))

        metrics = compute_all_metrics(results_by_lang, self.recall_ks, self.map_k)
        for lang, m in metrics.items():
            r1 = m.get("R@1", 0.0)
            logger.info("%s: R@1=%.4f n=%d", lang, r1, m.get("n_queries", 0))
        return {"metrics": metrics, "results_by_language": dict(results_by_lang)}
