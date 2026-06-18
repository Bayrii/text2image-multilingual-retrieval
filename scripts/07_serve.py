"""Step 7: launch FastAPI server."""
import _bootstrap  # noqa: F401
import uvicorn

from src.utils import load_config

if __name__ == "__main__":
    cfg = load_config()
    uvicorn.run(
        "src.api.main:app",
        host=cfg["api"]["host"],
        port=int(cfg["api"]["port"]),
        reload=False,
    )
