"""Common bootstrap: redirect caches, add project root to sys.path."""
import os
import sys
from pathlib import Path


def bootstrap():
    root = Path(r"E:\Coding\text2Image")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    os.environ.setdefault("HF_HOME", str(root / ".cache" / "huggingface"))
    os.environ.setdefault("TRANSFORMERS_CACHE", os.environ["HF_HOME"])
    os.environ.setdefault("HF_HUB_CACHE", os.environ["HF_HOME"])
    os.environ.setdefault("PIP_CACHE_DIR", str(root / ".cache" / "pip"))
    os.environ.setdefault("TORCH_HOME", str(root / ".cache" / "torch"))
    for v in ("HF_HOME", "TRANSFORMERS_CACHE", "HF_HUB_CACHE",
              "PIP_CACHE_DIR", "TORCH_HOME"):
        Path(os.environ[v]).mkdir(parents=True, exist_ok=True)


bootstrap()
