"""Step 1: download COCO val2017 captions + images."""
import _bootstrap  # noqa: F401
from src.data import download
from src.utils import load_config

if __name__ == "__main__":
    cfg = load_config()
    download.run(cfg)
