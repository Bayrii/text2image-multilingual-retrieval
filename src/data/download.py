"""Download MS-COCO val2017 captions + images.

Uses the val2017 captions JSON. Downloads only `max_images` images and
produces `captions_en.json`: a list of {image_id, file_path, language, caption}
with all 5 captions per image kept.
"""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Any, Dict, List

import requests
from PIL import Image
from tqdm import tqdm

from ..utils import ensure_dir, get_logger, load_config, save_json

logger = get_logger("download")

ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
IMG_URL_TPL = "http://images.cocodataset.org/val2017/{file_name}"


def _download_annotations(raw_dir: Path) -> Path:
    target = raw_dir / "captions_val2017.json"
    if target.exists():
        logger.info("Annotations already present: %s", target)
        return target
    ensure_dir(raw_dir)
    logger.info("Downloading COCO annotations zip (~250MB)...")
    with requests.get(ANN_URL, stream=True, timeout=120) as r:
        r.raise_for_status()
        buf = io.BytesIO()
        total = int(r.headers.get("Content-Length", 0))
        with tqdm(total=total, unit="B", unit_scale=True, desc="annotations") as pbar:
            for chunk in r.iter_content(chunk_size=1 << 15):
                buf.write(chunk)
                pbar.update(len(chunk))
        buf.seek(0)
        with zipfile.ZipFile(buf) as zf:
            with zf.open("annotations/captions_val2017.json") as src, open(target, "wb") as dst:
                dst.write(src.read())
    logger.info("Saved %s", target)
    return target


def _download_images(images_meta, images_dir: Path, max_images: int):
    ensure_dir(images_dir)
    selected = images_meta[:max_images]
    kept = []
    for img in tqdm(selected, desc="images"):
        file_path = images_dir / img["file_name"]
        if not file_path.exists():
            url = IMG_URL_TPL.format(file_name=img["file_name"])
            try:
                with requests.get(url, stream=True, timeout=60) as r:
                    r.raise_for_status()
                    with open(file_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=1 << 15):
                            f.write(chunk)
            except Exception as e:
                logger.warning("Failed download %s: %s", img["file_name"], e)
                if file_path.exists():
                    file_path.unlink(missing_ok=True)
                continue
        try:
            with Image.open(file_path) as im:
                im.verify()
        except Exception as e:
            logger.warning("Corrupt image %s: %s — removing", file_path, e)
            file_path.unlink(missing_ok=True)
            continue
        kept.append({"image_id": img["id"], "file_name": img["file_name"],
                     "file_path": str(file_path)})
    logger.info("Kept %d / %d images", len(kept), len(selected))
    return kept


def run(cfg: Dict[str, Any]) -> Path:
    raw_dir = Path(cfg["data"]["raw_dir"])
    images_dir = Path(cfg["data"]["images_dir"])
    processed_dir = Path(cfg["data"]["processed_dir"])
    out_path = Path(cfg["data"]["captions_en"])
    max_images = int(cfg["data"]["max_images"])

    ensure_dir(processed_dir)
    ann_path = _download_annotations(raw_dir)
    with open(ann_path, "r", encoding="utf-8") as f:
        ann = json.load(f)

    captions_by_image: Dict[int, List[str]] = {}
    for c in ann["annotations"]:
        captions_by_image.setdefault(c["image_id"], []).append(c["caption"].strip())

    images_meta = [im for im in ann["images"] if im["id"] in captions_by_image]
    images_meta.sort(key=lambda x: x["id"])

    kept = _download_images(images_meta, images_dir, max_images)

    rows = []
    for img in kept:
        for cap in captions_by_image.get(img["image_id"], []):
            rows.append({
                "image_id": img["image_id"],
                "file_path": img["file_path"],
                "language": "en",
                "caption": cap,
            })
    save_json(rows, out_path)
    logger.info("Wrote %d EN caption rows to %s", len(rows), out_path)
    return out_path


if __name__ == "__main__":
    from ..utils import setup_caches
    setup_caches()
    cfg = load_config()
    run(cfg)
