"""PyTorch Dataset for (image, caption) pairs + image_id-level splitting."""
from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from .augment import TextAugmenter

# CLIP ViT-B/32 image-processor constants (openai/clip-vit-base-patch32).
# Replicating these here lets the DataLoader avoid HF's image_processor, which
# allocates intermediate float64 numpy arrays on every sample — that hurts on
# RAM-constrained systems.
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def _build_clip_transform(image_size: int = 224) -> T.Compose:
    return T.Compose([
        T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(image_size),
        T.ToTensor(),  # uint8 PIL -> float32 [0,1] tensor
        T.Normalize(mean=_CLIP_MEAN, std=_CLIP_STD),
    ])


class COCOMultilingualDataset(Dataset):
    """Returns dict: image (Tensor 3xHxW), caption (str), image_id (int), language (str)."""

    def __init__(
        self,
        rows: List[Dict[str, Any]],
        image_processor,
        augmenter: Optional[TextAugmenter] = None,
        image_size: int = 224,
    ):
        self.rows = rows
        self.image_processor = image_processor  # kept for API compat / fallback
        self.augmenter = augmenter
        self.image_size = image_size
        self._tf = _build_clip_transform(image_size)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.rows[idx]
        img = Image.open(r["file_path"]).convert("RGB")
        pixel_values = self._tf(img)
        caption = r["caption"]
        if self.augmenter is not None:
            caption = self.augmenter(caption)
        return {
            "image": pixel_values,
            "caption": caption,
            "image_id": int(r["image_id"]),
            "language": r["language"],
        }


def split_by_image_id(
    rows: List[Dict[str, Any]],
    val_split: float,
    test_split: float,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Image-id level split: every caption for image X goes to one split."""
    image_ids = sorted({r["image_id"] for r in rows})
    rng = random.Random(seed)
    rng.shuffle(image_ids)
    n = len(image_ids)
    n_test = int(n * test_split)
    n_val = int(n * val_split)
    test_ids = set(image_ids[:n_test])
    val_ids = set(image_ids[n_test:n_test + n_val])
    train_ids = set(image_ids[n_test + n_val:])
    train, val, test = [], [], []
    for r in rows:
        if r["image_id"] in train_ids:
            train.append(r)
        elif r["image_id"] in val_ids:
            val.append(r)
        else:
            test.append(r)
    return train, val, test


class CollateFunction:
    """Tokenize captions and stack images. Pluggable text tokenizer (mCLIP-compatible)."""

    def __init__(self, text_tokenizer, max_length: int = 64):
        self.tok = text_tokenizer
        self.max_length = max_length

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        images = torch.stack([b["image"] for b in batch])
        captions = [b["caption"] for b in batch]
        image_ids = torch.tensor([b["image_id"] for b in batch], dtype=torch.long)
        languages = [b["language"] for b in batch]
        enc = self.tok(captions, padding=True, truncation=True,
                       max_length=self.max_length, return_tensors="pt")
        return {
            "images": images,
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "captions": captions,
            "image_ids": image_ids,
            "languages": languages,
        }


class HardNegativeDataset(Dataset):
    """Wrap base dataset; mining is done in the loss/trainer using batch embeddings."""

    def __init__(self, base: COCOMultilingualDataset):
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        return self.base[idx]
