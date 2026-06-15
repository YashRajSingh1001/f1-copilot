"""
Central config loaded from the project .env file.
Import get() everywhere instead of os.getenv() directly.
"""

import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def get(key: str, default: str = "") -> str:
    """Read a config value from .env, then process env, then default."""
    return os.getenv(key, default)
