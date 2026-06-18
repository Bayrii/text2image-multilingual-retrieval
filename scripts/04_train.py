"""Step 4: fine-tune mCLIP."""
import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
import torch
from torch.utils.data import DataLoader

from src.data.augment import TextAugmenter
from src.data.dataset import (COCOMultilingualDataset, CollateFunction,
                              split_by_image_id)
from src.models.mclip import MultilingualCLIPModel
from src.training.trainer import Trainer
from src.utils import get_device, get_logger, load_config, load_json

logger = get_logger("train")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).parent.parent /
                                                "config" / "config.yaml"))
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device()
    logger.info("Device: %s", device)

    rows = load_json(cfg["data"]["captions_multilingual"])
    train_rows, val_rows, _ = split_by_image_id(
        rows,
        val_split=float(cfg["data"]["val_split"]),
        test_split=float(cfg["data"]["test_split"]),
    )
    logger.info("Train rows=%d val rows=%d", len(train_rows), len(val_rows))

    model = MultilingualCLIPModel(cfg)

    train_ds = COCOMultilingualDataset(
        train_rows,
        image_processor=model.image_processor,
        augmenter=TextAugmenter(seed=42),
        image_size=cfg["data"]["image_size"],
    )
    val_ds = COCOMultilingualDataset(
        val_rows,
        image_processor=model.image_processor,
        augmenter=None,
        image_size=cfg["data"]["image_size"],
    )

    collate = CollateFunction(model.text_tokenizer, max_length=64)
    nw = int(cfg["training"].get("num_workers", 0))
    bs = int(cfg["training"]["batch_size"])
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=nw, collate_fn=collate, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False,
                            num_workers=nw, collate_fn=collate, pin_memory=True)

    trainer = Trainer(model, train_loader, val_loader, cfg, device)
    if args.resume:
        ep, met = trainer.load_checkpoint(args.resume)
        logger.info("Resumed from %s (epoch=%d metric=%.4f)", args.resume, ep, met)
    trainer.train()
    logger.info("Best val R@1: %.4f", trainer.best_recall)


if __name__ == "__main__":
    main()
