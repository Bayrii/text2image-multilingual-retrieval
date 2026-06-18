"""Translate EN COCO captions to all target languages.

Uses Helsinki-NLP/opus-mt-en-{lang} models, batched, with incremental save.
"""
from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

from ..utils import get_logger, load_config, load_json, save_json

logger = get_logger("translate")

OPUS_LANG_MAP = {
    "ar": "ar", "zh": "zh", "fr": "fr", "de": "de", "ru": "ru",
    "es": "es", "it": "it", "ja": "jap", "tr": "trk", "az": "az", "sw": "sw",
}


def _load_translator(target_lang: str):
    import torch
    from transformers import MarianMTModel, MarianTokenizer
    suffix = OPUS_LANG_MAP.get(target_lang, target_lang)
    name = f"Helsinki-NLP/opus-mt-en-{suffix}"
    logger.info("Loading %s ...", name)
    tok = MarianTokenizer.from_pretrained(name)
    model = MarianMTModel.from_pretrained(name)
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()
    return tok, model


def _translate_batch(texts, tok, model, batch_size: int = 32):
    import torch
    out = []
    device = next(model.parameters()).device
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        with torch.no_grad():
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                      max_length=128).to(device)
            gen = model.generate(**enc, max_length=160, num_beams=2)
            out.extend(tok.batch_decode(gen, skip_special_tokens=True))
    return out


def run(cfg: Dict[str, Any]) -> Path:
    import torch
    in_path = Path(cfg["data"]["captions_en"])
    out_path = Path(cfg["data"]["captions_multilingual"])
    languages = list(cfg["data"]["languages"])

    rows_en = load_json(in_path)
    logger.info("Loaded %d EN rows", len(rows_en))

    if out_path.exists():
        existing = load_json(out_path)
        done_langs = {r["language"] for r in existing}
        logger.info("Resuming. Already have languages: %s", sorted(done_langs))
    else:
        existing = list(rows_en)
        done_langs = {"en"}
        save_json(existing, out_path)

    for lang in languages:
        if lang in done_langs:
            continue
        try:
            tok, model = _load_translator(lang)
        except Exception as e:
            logger.warning("Skipping %s -- model unavailable: %s", lang, e)
            continue

        captions_en = [r["caption"] for r in rows_en]
        translated = _translate_batch(captions_en, tok, model)
        new_rows = []
        for r, txt in zip(rows_en, translated):
            new_rows.append({
                "image_id": r["image_id"],
                "file_path": r["file_path"],
                "language": lang,
                "caption": txt,
            })
        existing.extend(new_rows)
        save_json(existing, out_path)
        logger.info("Added %d %s rows (total=%d)", len(new_rows), lang, len(existing))

        del model, tok
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info("Wrote %d multilingual rows to %s", len(existing), out_path)
    return out_path


if __name__ == "__main__":
    from ..utils import setup_caches
    setup_caches()
    cfg = load_config()
    run(cfg)
