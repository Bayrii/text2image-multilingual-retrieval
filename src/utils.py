"""Shared utilities: cache redirection, config loader, logging, device selection."""
import os
import sys
import json
import logging
from pathlib import Path
from typing import Any, Dict


def setup_caches() -> None:
    """Redirect HF / pip / torch caches to E: drive. Call at the top of every script."""
    root = r"E:\Coding\text2Image"
    os.environ.setdefault("HF_HOME", os.path.join(root, ".cache", "huggingface"))
    os.environ.setdefault("TRANSFORMERS_CACHE", os.environ["HF_HOME"])
    os.environ.setdefault("HF_HUB_CACHE", os.environ["HF_HOME"])
    os.environ.setdefault("PIP_CACHE_DIR", os.path.join(root, ".cache", "pip"))
    os.environ.setdefault("TORCH_HOME", os.path.join(root, ".cache", "torch"))
    for v in ("HF_HOME", "TRANSFORMERS_CACHE", "HF_HUB_CACHE", "PIP_CACHE_DIR", "TORCH_HOME"):
        Path(os.environ[v]).mkdir(parents=True, exist_ok=True)


def load_config(path: str | os.PathLike = None) -> Dict[str, Any]:
    """Load YAML config. Defaults to <project_root>/config/config.yaml."""
    import yaml
    if path is None:
        path = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_logger(name: str = "t2i", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                                     datefmt="%H:%M:%S"))
    logger.addHandler(h)
    return logger


def get_device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_dir(p: str | os.PathLike) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: Any, path: str | os.PathLike) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str | os.PathLike) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
