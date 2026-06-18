"""Training loop: gradient accumulation, mixed precision, R@1 eval, early stopping."""
from __future__ import annotations

import csv
import gc
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from ..models.losses import ContrastiveWithHardNegatives, InfoNCELoss
from ..utils import ensure_dir, get_logger
from .scheduler import warmup_cosine_schedule

logger = get_logger("trainer")


class Trainer:
    def __init__(self, model, train_loader, val_loader, config: Dict[str, Any],
                 device: torch.device):
        self.model = model.to(device)
        # On RAM-constrained Windows, release CPU shadow copies of weights now
        # that the model lives on the GPU.  Otherwise ~3 GB stays committed in
        # the page file and small allocations later (image preprocessing) fail.
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        tcfg = config["training"]
        self.batch_size = int(tcfg["batch_size"])
        self.lr = float(tcfg["learning_rate"])
        self.warmup = int(tcfg["warmup_steps"])
        self.max_epochs = int(tcfg["max_epochs"])
        self.accum = int(tcfg["gradient_accumulation_steps"])
        self.use_amp = bool(tcfg["mixed_precision"]) and device.type == "cuda"
        self.temperature = float(tcfg["temperature"])
        self.hn_k = int(tcfg["hard_negatives_per_sample"])
        self.early_stop_patience = int(tcfg["early_stopping_patience"])
        self.checkpoint_every = int(tcfg["checkpoint_every_n_epochs"])
        self.weight_decay = float(tcfg.get("weight_decay", 0.01))
        self.max_grad_norm = float(tcfg.get("max_grad_norm", 1.0))
        self.log_every = int(tcfg.get("log_every_n_steps", 20))
        self.ckpt_dir = ensure_dir(tcfg["checkpoints_dir"])
        self.log_csv = Path(tcfg["log_csv"])

        self.criterion = ContrastiveWithHardNegatives(
            temperature=self.temperature, hn_k=self.hn_k
        )

        params = self.model.trainable_parameters()
        self.optimizer = torch.optim.AdamW(params, lr=self.lr,
                                           weight_decay=self.weight_decay)
        steps_per_epoch = max(1, len(train_loader) // self.accum)
        total_steps = steps_per_epoch * self.max_epochs
        self.scheduler = warmup_cosine_schedule(self.optimizer, self.warmup, total_steps)
        self.scaler = GradScaler(enabled=self.use_amp)

        self.global_step = 0
        self.best_recall = 0.0
        self.best_val = float("inf")
        self.epochs_since_improve = 0
        self._init_log()

    def _init_log(self):
        self.log_csv.parent.mkdir(parents=True, exist_ok=True)
        if not self.log_csv.exists():
            with open(self.log_csv, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(["epoch", "step", "train_loss",
                                        "val_loss", "val_recall_at_1", "lr"])

    def _log_row(self, epoch, step, train_loss, val_loss, val_r1, lr):
        with open(self.log_csv, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch, step, train_loss, val_loss, val_r1, lr])

    def train(self):
        for epoch in range(1, self.max_epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss, val_r1 = self.validate()
            lr = self.optimizer.param_groups[0]["lr"]
            logger.info("[epoch %d] train_loss=%.4f val_loss=%.4f val_R@1=%.4f lr=%.2e",
                        epoch, train_loss, val_loss, val_r1, lr)
            self._log_row(epoch, self.global_step, train_loss, val_loss, val_r1, lr)

            improved = val_r1 > self.best_recall
            if improved:
                self.best_recall = val_r1
                self.best_val = val_loss
                self.epochs_since_improve = 0
                self.save_checkpoint(self.ckpt_dir / "best.pt", epoch=epoch, metric=val_r1)
            else:
                self.epochs_since_improve += 1

            if epoch % self.checkpoint_every == 0:
                self.save_checkpoint(self.ckpt_dir / f"epoch_{epoch}.pt",
                                     epoch=epoch, metric=val_r1)
            self.save_checkpoint(self.ckpt_dir / "latest.pt", epoch=epoch, metric=val_r1)

            if self.epochs_since_improve >= self.early_stop_patience:
                logger.info("Early stopping at epoch %d (no improvement %d epochs)",
                            epoch, self.early_stop_patience)
                break

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        running_loss = 0.0
        running_n = 0
        t0 = time.time()
        self.optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(self.train_loader, desc=f"train e{epoch}")
        for step, batch in enumerate(pbar):
            images = batch["images"].to(self.device, non_blocking=True)
            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            attn = batch["attention_mask"].to(self.device, non_blocking=True)
            ids = batch["image_ids"].to(self.device, non_blocking=True)

            with autocast(enabled=self.use_amp):
                img_emb, txt_emb = self.model(images, input_ids, attn)
                loss = self.criterion(img_emb, txt_emb, ids) / self.accum

            self.scaler.scale(loss).backward()

            if (step + 1) % self.accum == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.trainable_parameters(),
                                               self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1

            bs = images.size(0)
            running_loss += loss.item() * self.accum * bs
            running_n += bs
            if (step + 1) % self.log_every == 0:
                avg = running_loss / max(1, running_n)
                cur_lr = self.optimizer.param_groups[0]["lr"]
                pbar.set_postfix(loss=f"{avg:.4f}", lr=f"{cur_lr:.2e}")

        elapsed = time.time() - t0
        avg = running_loss / max(1, running_n)
        logger.info("epoch %d done loss=%.4f (%.1fs)", epoch, avg, elapsed)
        return avg

    @torch.no_grad()
    def validate(self) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        total_n = 0
        all_img_emb = []
        all_txt_emb = []
        all_ids = []
        for batch in tqdm(self.val_loader, desc="val"):
            images = batch["images"].to(self.device)
            input_ids = batch["input_ids"].to(self.device)
            attn = batch["attention_mask"].to(self.device)
            ids = batch["image_ids"].to(self.device)
            with autocast(enabled=self.use_amp):
                img_emb, txt_emb = self.model(images, input_ids, attn)
                loss = InfoNCELoss(self.temperature)(img_emb, txt_emb, ids)
            total_loss += loss.item() * images.size(0)
            total_n += images.size(0)
            all_img_emb.append(img_emb.float().cpu())
            all_txt_emb.append(txt_emb.float().cpu())
            all_ids.append(ids.cpu())

        if total_n == 0:
            return 0.0, 0.0
        img_e = torch.cat(all_img_emb)
        txt_e = torch.cat(all_txt_emb)
        ids = torch.cat(all_ids)

        # Build unique-image gallery: first occurrence of each image_id
        seen = {}
        gal_idx = []
        gal_ids = []
        for i, iid in enumerate(ids.tolist()):
            if iid not in seen:
                seen[iid] = i
                gal_idx.append(i)
                gal_ids.append(iid)
        gal = img_e[gal_idx]
        gal_ids_t = torch.tensor(gal_ids)
        sim = txt_e @ gal.t()
        top1 = sim.argmax(dim=1)
        pred_ids = gal_ids_t[top1]
        recall_at_1 = (pred_ids == ids).float().mean().item()
        return total_loss / total_n, recall_at_1

    def save_checkpoint(self, path: Path, epoch: int, metric: float):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "epoch": epoch,
            "metric": metric,
            "config": self.config,
        }, path)
        logger.info("Saved %s (epoch=%d metric=%.4f)", path, epoch, metric)

    def load_checkpoint(self, path: Path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        return ckpt.get("epoch", 0), ckpt.get("metric", 0.0)
