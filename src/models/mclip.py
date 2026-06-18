"""mCLIP wrapper: multilingual XLM-R text encoder + CLIP visual encoder.

Both encoders project to a shared 512-d space and outputs are L2-normalized.
Supports partial layer freezing + gradient checkpointing to fit on small GPUs.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


def _l2norm(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return F.normalize(x, p=2, dim=dim, eps=1e-8)


class MultilingualCLIPModel(nn.Module):
    """Combined text + visual model. text_encoder is M-CLIP (XLM-R + projection).

    Visual encoder is HuggingFace CLIP ViT (its `get_image_features` already
    projects to the shared 512-d space).
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        from multilingual_clip import pt_multilingual_clip
        from transformers import AutoTokenizer, CLIPModel, CLIPProcessor

        text_name = config["model"]["mclip_model"]
        visual_name = config["model"]["clip_visual"]

        self.text_encoder = pt_multilingual_clip.MultilingualCLIP.from_pretrained(text_name)
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_name)

        self.visual_full = CLIPModel.from_pretrained(visual_name)
        self.image_processor = CLIPProcessor.from_pretrained(visual_name)

        self.embedding_dim = int(config["model"]["embedding_dim"])

        if config["model"].get("freeze_visual_encoder", True):
            for p in self.visual_full.parameters():
                p.requires_grad = False

        freeze_below = int(config["model"].get("freeze_text_layers_below", 0))
        if freeze_below > 0:
            self._freeze_text_layers_below(freeze_below)

        if config["model"].get("gradient_checkpointing", False):
            self._enable_gradient_checkpointing()

    def _freeze_text_layers_below(self, n: int) -> None:
        """Freeze XLM-R embeddings + transformer layers 0..n-1.

        Keeps top (24 - n) layers and the LinearTransformation projection trainable.
        """
        layer_re = re.compile(r"(?:^|\.)encoder\.layer\.(\d+)\.")
        frozen = 0
        for name, p in self.text_encoder.transformer.named_parameters():
            if "embeddings." in name:
                p.requires_grad = False
                frozen += 1
                continue
            m = layer_re.search(name)
            if m and int(m.group(1)) < n:
                p.requires_grad = False
                frozen += 1

    def _enable_gradient_checkpointing(self) -> None:
        """Trade compute for memory by recomputing activations during backward."""
        try:
            self.text_encoder.transformer.gradient_checkpointing_enable()
        except Exception:
            pass
        # HF >=4.40 also disables use_cache when checkpointing; mirror that.
        if hasattr(self.text_encoder.transformer, "config"):
            try:
                self.text_encoder.transformer.config.use_cache = False
            except Exception:
                pass

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        device = self.device
        enc = self.text_tokenizer(texts, padding=True, truncation=True,
                                  max_length=64, return_tensors="pt").to(device)
        embs = self.text_encoder.transformer(**enc)[0]
        att = enc["attention_mask"]
        embs = (embs * att.unsqueeze(2)).sum(dim=1) / att.sum(dim=1, keepdim=True).clamp(min=1)
        embs = self.text_encoder.LinearTransformation(embs)
        return _l2norm(embs)

    def encode_text_tokenized(self, input_ids: torch.Tensor,
                              attention_mask: torch.Tensor) -> torch.Tensor:
        embs = self.text_encoder.transformer(input_ids=input_ids,
                                             attention_mask=attention_mask)[0]
        att = attention_mask
        embs = (embs * att.unsqueeze(2)).sum(dim=1) / att.sum(dim=1, keepdim=True).clamp(min=1)
        embs = self.text_encoder.LinearTransformation(embs)
        return _l2norm(embs)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.visual_full.get_image_features(pixel_values=images.to(self.device))
        return _l2norm(feats)

    def forward(self, images: torch.Tensor, input_ids: torch.Tensor,
                attention_mask: torch.Tensor):
        img_emb = self.encode_image(images)
        txt_emb = self.encode_text_tokenized(input_ids, attention_mask)
        return img_emb, txt_emb

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
