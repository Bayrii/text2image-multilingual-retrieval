"""Step 8: pretrained-vs-fine-tuned comparison.

Re-evaluates the pretrained mCLIP and the fine-tuned checkpoint on the
SAME test split, generates a side-by-side report at
    reports/comparison/comparison.md  (+ comparison_metrics.json + 4 plots)
"""
import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from src.data.dataset import split_by_image_id
from src.evaluation.compare import run_comparison
from src.utils import get_device, get_logger, load_config, load_json

logger = get_logger("compare_models")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", default=None,
                        help="Path to fine-tuned checkpoint (default: checkpoints/best.pt)")
    parser.add_argument("--output", default=None,
                        help="Output dir (default: reports/comparison/)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device()
    logger.info("Device: %s", device)

    ckpt_path = Path(args.checkpoint) if args.checkpoint else \
        Path(cfg["training"]["checkpoints_dir"]) / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No fine-tuned checkpoint at {ckpt_path}. Run scripts/04_train.py first."
        )

    out_dir = Path(args.output) if args.output else \
        Path(cfg["evaluation"]["reports_dir"]) / "comparison"

    rows = load_json(cfg["data"]["captions_multilingual"])
    _, _, test = split_by_image_id(
        rows,
        val_split=float(cfg["data"]["val_split"]),
        test_split=float(cfg["data"]["test_split"]),
    )
    logger.info("All rows: %d  |  Test rows: %d  |  Languages: %d",
                len(rows), len(test), len({r["language"] for r in test}))

    run_comparison(cfg, rows, test, device, ckpt_path, out_dir)
    logger.info("Done. Open: %s", out_dir / "comparison.md")


if __name__ == "__main__":
    main()
