"""InfoNCE / NT-Xent contrastive loss + in-batch hard negative mining."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """Symmetric image<->text contrastive loss.

    Both directions averaged: image-as-query and text-as-query.
    Identical-image-id rows in a batch are masked out as positives, so when
    multiple captions for the same image co-exist in a batch, they don't
    pollute the negatives.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, image_emb: torch.Tensor, text_emb: torch.Tensor,
                image_ids: torch.Tensor | None = None) -> torch.Tensor:
        logits = image_emb @ text_emb.t() / self.temperature
        n = logits.size(0)
        targets = torch.arange(n, device=logits.device)

        if image_ids is not None:
            same = image_ids.unsqueeze(0) == image_ids.unsqueeze(1)
            mask = same & ~torch.eye(n, dtype=torch.bool, device=logits.device)
            logits_i2t = logits.masked_fill(mask, float("-inf"))
            logits_t2i = logits.t().masked_fill(mask, float("-inf"))
        else:
            logits_i2t = logits
            logits_t2i = logits.t()

        loss_i2t = F.cross_entropy(logits_i2t, targets)
        loss_t2i = F.cross_entropy(logits_t2i, targets)
        return 0.5 * (loss_i2t + loss_t2i)


class HardNegativeMiner:
    """Identify the top-k hardest in-batch negatives per sample.

    A "hard negative" for sample i is sample j (j != i, image_id[j] != image_id[i])
    with the highest text-embedding similarity to text[i].
    Returns indices [B, k].
    """

    def __init__(self, k: int = 3):
        self.k = k

    def __call__(self, text_emb: torch.Tensor,
                 image_ids: torch.Tensor) -> torch.Tensor:
        n = text_emb.size(0)
        sim = text_emb @ text_emb.t()
        same = image_ids.unsqueeze(0) == image_ids.unsqueeze(1)
        sim = sim.masked_fill(same, float("-inf"))
        k = min(self.k, n - 1)
        if k <= 0:
            return torch.empty(n, 0, dtype=torch.long, device=text_emb.device)
        _, idx = sim.topk(k, dim=1)
        return idx


class ContrastiveWithHardNegatives(nn.Module):
    """InfoNCE plus an extra triplet-style margin term over mined hard negatives."""

    def __init__(self, temperature: float = 0.07, hn_weight: float = 0.2,
                 hn_margin: float = 0.2, hn_k: int = 3):
        super().__init__()
        self.base = InfoNCELoss(temperature)
        self.miner = HardNegativeMiner(hn_k)
        self.hn_weight = hn_weight
        self.hn_margin = hn_margin

    def forward(self, image_emb: torch.Tensor, text_emb: torch.Tensor,
                image_ids: torch.Tensor) -> torch.Tensor:
        base_loss = self.base(image_emb, text_emb, image_ids)
        hn_idx = self.miner(text_emb, image_ids)
        if hn_idx.numel() == 0:
            return base_loss
        pos_sim = (image_emb * text_emb).sum(dim=1, keepdim=True)
        neg_text = text_emb[hn_idx]
        neg_sim = (image_emb.unsqueeze(1) * neg_text).sum(dim=2)
        margin_loss = F.relu(self.hn_margin - pos_sim + neg_sim).mean()
        return base_loss + self.hn_weight * margin_loss
