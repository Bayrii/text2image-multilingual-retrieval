"""Text augmentation for training.

Random lowercase/uppercase first letter, punctuation removal, word dropout.
Back-translation augmentation is provided but NOT applied online (too slow)
-- generate it offline and add to the dataset if desired.
"""
from __future__ import annotations

import random
import string
from typing import List, Optional


class TextAugmenter:
    def __init__(
        self,
        lower_p: float = 0.2,
        upper_first_p: float = 0.2,
        strip_punct_p: float = 0.2,
        word_dropout_p: float = 0.1,
        seed: Optional[int] = None,
    ):
        self.lower_p = lower_p
        self.upper_first_p = upper_first_p
        self.strip_punct_p = strip_punct_p
        self.word_dropout_p = word_dropout_p
        self.rng = random.Random(seed)

    def __call__(self, text: str) -> str:
        if not text:
            return text
        if self.rng.random() < self.lower_p:
            text = text.lower()
        elif self.rng.random() < self.upper_first_p and text:
            text = text[0].upper() + text[1:]
        if self.rng.random() < self.strip_punct_p:
            text = text.translate(str.maketrans("", "", string.punctuation))
        if self.word_dropout_p > 0:
            words = text.split()
            if len(words) > 3:
                words = [w for w in words if self.rng.random() >= self.word_dropout_p]
                if words:
                    text = " ".join(words)
        return text


def back_translate(texts: List[str], src: str = "en", pivot: str = "fr") -> List[str]:
    """Offline back-translation: en -> pivot -> en, used to generate paraphrases."""
    from transformers import MarianMTModel, MarianTokenizer
    import torch

    def _trans(model_name, batch):
        tok = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        if torch.cuda.is_available():
            model = model.to("cuda")
        device = next(model.parameters()).device
        out = []
        with torch.no_grad():
            for i in range(0, len(batch), 32):
                enc = tok(batch[i:i + 32], return_tensors="pt", padding=True,
                          truncation=True, max_length=128).to(device)
                gen = model.generate(**enc, max_length=160, num_beams=2)
                out.extend(tok.batch_decode(gen, skip_special_tokens=True))
        del model, tok
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return out

    forward = _trans(f"Helsinki-NLP/opus-mt-{src}-{pivot}", texts)
    back = _trans(f"Helsinki-NLP/opus-mt-{pivot}-{src}", forward)
    return back
