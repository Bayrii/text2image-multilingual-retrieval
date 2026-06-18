"""Step 6: evaluate the model on the test split, generate report.

Compares two retrievers (two_stage on the image index, text_bridge on the
text index) and writes per-retriever subdirectories.
"""
import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
import torch

from src.data.dataset import split_by_image_id
from src.evaluation.evaluator import Evaluator
from src.evaluation.report import generate_report
from src.models.mclip import MultilingualCLIPModel
from src.retrieval.indexer import FAISSIndexer
from src.retrieval.retriever import TextBridgeRetriever, TwoStageRetriever
from src.utils import get_device, get_logger, load_config, load_json

logger = get_logger("evaluate")


def _filter_test_rows(rows, test_image_ids):
    s = set(test_image_ids)
    return [r for r in rows if int(r["image_id"]) in s]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device()
    out_dir = Path(args.output) if args.output else \
        Path(cfg["evaluation"]["reports_dir"]) / "eval_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_json(cfg["data"]["captions_multilingual"])
    _, _, test = split_by_image_id(
        rows,
        val_split=float(cfg["data"]["val_split"]),
        test_split=float(cfg["data"]["test_split"]),
    )
    logger.info("Test rows: %d (across %d languages)", len(test),
                len({r["language"] for r in test}))

    model = MultilingualCLIPModel(cfg).to(device)
    ckpt_path = Path(args.checkpoint) if args.checkpoint else \
        Path(cfg["training"]["checkpoints_dir"]) / "best.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"], strict=False)
        logger.info("Loaded checkpoint %s", ckpt_path)
    else:
        logger.warning("No checkpoint found at %s; evaluating pretrained model",
                       ckpt_path)
    model.eval()

    image_indexer = FAISSIndexer(dim=int(cfg["model"]["embedding_dim"]),
                                 index_type=cfg["retrieval"]["faiss_index_type"])
    image_indexer.load(cfg["retrieval"]["index_path"],
                       cfg["retrieval"]["meta_path"],
                       emb_path=cfg["retrieval"]["image_emb_path"])

    text_indexer = FAISSIndexer(dim=int(cfg["model"]["embedding_dim"]),
                                index_type=cfg["retrieval"]["faiss_index_type"])
    text_indexer.load(cfg["retrieval"]["text_index_path"],
                      cfg["retrieval"]["text_meta_path"],
                      emb_path=cfg["retrieval"]["text_emb_path"])

    notes_two = "Retriever: **two-stage** (FAISS image index + exact re-rank)."
    notes_tb = "Retriever: **text-bridge** (caption-index match, then group by image)."

    two_stage = TwoStageRetriever(model, image_indexer, cfg)
    tb = TextBridgeRetriever(model, text_indexer, cfg)

    logger.info("=== Evaluating two_stage ===")
    out2 = Evaluator(two_stage, test, cfg).run()
    generate_report(out2, out_dir / "two_stage", notes=notes_two)

    logger.info("=== Evaluating text_bridge ===")
    out_tb = Evaluator(tb, test, cfg).run()
    generate_report(out_tb, out_dir / "text_bridge", notes=notes_tb)

    overall_two = out2["metrics"]["overall"]
    overall_tb = out_tb["metrics"]["overall"]
    logger.info("two_stage   overall R@1=%.4f R@5=%.4f", overall_two["R@1"], overall_two["R@5"])
    logger.info("text_bridge overall R@1=%.4f R@5=%.4f", overall_tb["R@1"], overall_tb["R@5"])


if __name__ == "__main__":
    main()
