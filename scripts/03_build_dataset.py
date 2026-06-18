"""Step 3: build CSV + Parquet + train/val/test split JSON (image-id-level)."""
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd

from src.data.dataset import split_by_image_id
from src.utils import get_logger, load_config, load_json, save_json

logger = get_logger("build_dataset")


def main():
    cfg = load_config()
    rows = load_json(cfg["data"]["captions_multilingual"])
    df = pd.DataFrame(rows)
    csv_p = Path(cfg["data"]["dataset_csv"])
    pq_p = Path(cfg["data"]["dataset_parquet"])
    csv_p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_p, index=False, encoding="utf-8")
    df.to_parquet(pq_p, index=False)
    logger.info("Wrote %d rows -> %s and %s", len(df), csv_p, pq_p)

    train, val, test = split_by_image_id(
        rows,
        val_split=float(cfg["data"]["val_split"]),
        test_split=float(cfg["data"]["test_split"]),
    )
    splits = {
        "train_image_ids": sorted({r["image_id"] for r in train}),
        "val_image_ids": sorted({r["image_id"] for r in val}),
        "test_image_ids": sorted({r["image_id"] for r in test}),
    }
    save_json(splits, cfg["data"]["splits_json"])
    logger.info("Splits: train=%d val=%d test=%d (image_ids)",
                len(splits["train_image_ids"]),
                len(splits["val_image_ids"]),
                len(splits["test_image_ids"]))
    logger.info("Splits: train=%d val=%d test=%d (rows)",
                len(train), len(val), len(test))


if __name__ == "__main__":
    main()
