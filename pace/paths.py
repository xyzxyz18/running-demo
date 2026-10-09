"""Shared project and runtime-data paths."""
import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent


def data_dir() -> Path:
    """Honor PACE_DATA_DIR, otherwise reuse the project output directory."""
    return Path(os.environ.get("PACE_DATA_DIR", str(PROJECT_DIR / "output"))).resolve()
