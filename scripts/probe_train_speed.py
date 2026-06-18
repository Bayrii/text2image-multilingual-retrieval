"""Time ~30 training batches and print per-epoch + total ETA.

Mirrors scripts/04_train.py setup so the measurement is realistic.
"""
import argparse
import time
from pathlib import Path

import _bootstrap  # noqa: F401
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from src.data.augment import TextAugmenter
from src.data.dataset import (COCOMultilingualDataset, CollateFunction,
                              split_by_image_id)
from src.models.losses import ContrastiveWithHardNegatives
from src.models.mclip import MultilingualCLIPModel
from src.utils import get_device, get_logger, load_config, load_json

logger = get_logger("probe")


def fmt_hms(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}h{m:02d}m{s:02d}s"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(Path(__file__).parent.parent /
                                           "config" / "config.yaml"))
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--measure", type=int, default=25)
    args = p.parse_args()

    cfg = load_config(args.config)
    device = get_device()
    logger.info("Device: %s", device)

    rows = load_json(cfg["data"]["captions_multilingual"])
    train_rows, val_rows, _ = split_by_image_id(
        rows,
        val_split=float(cfg["data"]["val_split"]),
        test_split=float(cfg["data"]["test_split"]),
    )
    logger.info("Train rows=%d", len(train_rows))

    t0 = time.time()
    model = MultilingualCLIPModel(cfg).to(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model loaded in %.1fs  trainable_params=%.1fM",
                time.time() - t0, n_train / 1e6)

    train_ds = COCOMultilingualDataset(
        train_rows,
        image_processor=model.image_processor,
        augmenter=TextAugmenter(seed=42),
        image_size=cfg["data"]["image_size"],
    )
    collate = CollateFunction(model.text_tokenizer, max_length=64)
    bs = int(cfg["training"]["batch_size"])
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=int(cfg["training"].get("num_workers", 0)),
                              collate_fn=collate, pin_memory=True)
    n_batches = len(train_loader)

    tcfg = cfg["training"]
    use_amp = bool(tcfg["mixed_precision"]) and device.type == "cuda"
    accum = int(tcfg["gradient_accumulation_steps"])
    temperature = float(tcfg["temperature"])
    hn_k = int(tcfg["hard_negatives_per_sample"])
    lr = float(tcfg["learning_rate"])
    weight_decay = float(tcfg.get("weight_decay", 0.01))

    criterion = ContrastiveWithHardNegatives(temperature=temperature, hn_k=hn_k)
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=lr,
                                  weight_decay=weight_decay)
    scaler = GradScaler(enabled=use_amp)

    total = args.warmup + args.measure
    logger.info("Probing: %d warmup + %d measured batches  (bs=%d, accum=%d, %d batches/epoch)",
                args.warmup, args.measure, bs, accum, n_batches)

    model.train()
    t_start = None
    seen = 0
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(train_loader):
        if step == args.warmup:
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_start = time.time()
        if step >= total:
            break

        images = batch["images"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attn = batch["attention_mask"].to(device, non_blocking=True)
        ids = batch["image_ids"].to(device, non_blocking=True)
        with autocast(enabled=use_amp):
            img_emb, txt_emb = model(images, input_ids, attn)
            loss = criterion(img_emb, txt_emb, ids) / accum
        scaler.scale(loss).backward()
        if (step + 1) % accum == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        if step >= args.warmup:
            seen += 1

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t_start
    sec_per_batch = elapsed / max(1, seen)
    sec_per_epoch = sec_per_batch * n_batches

    logger.info("=" * 64)
    logger.info("RESULT  %.3f sec/batch over %d measured batches", sec_per_batch, seen)
    logger.info("        ~%s per epoch  (%d batches)",
                fmt_hms(sec_per_epoch), n_batches)
    for ep in (1, 2, 3, 5, 10):
        logger.info("        %2d epochs -> ~%s", ep, fmt_hms(sec_per_epoch * ep))

    if device.type == "cuda":
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        logger.info("        peak VRAM: %.2f GB", peak_gb)


if __name__ == "__main__":
    main()
