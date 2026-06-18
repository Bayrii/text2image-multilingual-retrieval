"""Step 2: translate captions to all target languages."""
import _bootstrap  # noqa: F401
from src.data import translate
from src.utils import load_config

if __name__ == "__main__":
    cfg = load_config()
    translate.run(cfg)
