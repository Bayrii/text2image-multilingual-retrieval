"""Step 5: embed all images + captions, build FAISS indexes."""
import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.mclip import MultilingualCLIPModel
from src.retrieval.indexer import FAISSIndexer
from src.utils import get_device, get_logger, load_config, load_json

logger = get_logger("build_index")


def _load_finetuned(model: MultilingualCLIPModel, ckpt_path: Path, device):
    if ckpt_path and ckpt_path.exists():
        logger.info("Loading checkpoint: %s", ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"], strict=False)
    else:
        logger.warning("No checkpoint at %s; using pretrained weights", ckpt_path)
    return model


@torch.no_grad()
def embed_images(model, file_paths, batch_size=32, device=torch.device("cpu")):
    embs = []
    for i in tqdm(range(0, len(file_paths), batch_size), desc="image-embed"):
        chunk = file_paths[i:i + batch_size]
        pil = [Image.open(p).convert("RGB") for p in chunk]
        pixels = model.image_processor(images=pil, return_tensors="pt")["pixel_values"].to(device)
        e = model.encode_image(pixels).float().cpu().numpy()
        embs.append(e)
    return np.concatenate(embs, axis=0)


@torch.no_grad()
def embed_texts(model, texts, batch_size=64, device=torch.device("cpu")):
    embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="text-embed"):
        chunk = texts[i:i + batch_size]
        e = model.encode_text(chunk).float().cpu().numpy()
        embs.append(e)
    return np.concatenate(embs, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", default=None,
                        help="Path to fine-tuned checkpoint. Defaults to best.pt.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device()
    logger.info("Device: %s", device)

    model = MultilingualCLIPModel(cfg).to(device)
    ckpt = Path(args.checkpoint) if args.checkpoint else \
        Path(cfg["training"]["checkpoints_dir"]) / "best.pt"
    _load_finetuned(model, ckpt, device)
    model.eval()

    rows = load_json(cfg["data"]["captions_multilingual"])

    # Image index: one entry per unique image
    seen = {}
    img_meta = []
    img_files = []
    for r in rows:
        iid = int(r["image_id"])
        if iid in seen:
            continue
        seen[iid] = True
        img_meta.append({"image_id": iid, "file_path": r["file_path"]})
        img_files.append(r["file_path"])
    logger.info("Embedding %d unique images...", len(img_files))
    img_embs = embed_images(model, img_files, device=device)

    image_indexer = FAISSIndexer(dim=int(cfg["model"]["embedding_dim"]),
                                 index_type=cfg["retrieval"]["faiss_index_type"])
    image_indexer.build(img_embs, img_meta)
    image_indexer.save(cfg["retrieval"]["index_path"],
                       cfg["retrieval"]["meta_path"],
                       emb_path=cfg["retrieval"]["image_emb_path"])

    # Text index: one entry per (image_id, language, caption)
    text_meta = [{
        "image_id": int(r["image_id"]),
        "file_path": r["file_path"],
        "language": r["language"],
        "caption": r["caption"],
    } for r in rows]
    captions = [r["caption"] for r in rows]
    logger.info("Embedding %d captions...", len(captions))
    txt_embs = embed_texts(model, captions, device=device)

    text_indexer = FAISSIndexer(dim=int(cfg["model"]["embedding_dim"]),
                                index_type=cfg["retrieval"]["faiss_index_type"])
    text_indexer.build(txt_embs, text_meta)
    text_indexer.save(cfg["retrieval"]["text_index_path"],
                      cfg["retrieval"]["text_meta_path"],
                      emb_path=cfg["retrieval"]["text_emb_path"])
    logger.info("Done. image_index=%d text_index=%d",
                image_indexer.size, text_indexer.size)


if __name__ == "__main__":
    main()
