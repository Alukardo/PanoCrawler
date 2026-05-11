import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
import yaml

PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

def load_project_env() -> bool:
    return load_dotenv(PROJECT_ROOT / ".apikey")


load_project_env()

with open(CONFIG_PATH, encoding="utf-8") as f:
    cfg: dict[str, Any] = yaml.safe_load(f) or {}


def resolve_project_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def get_images_root() -> Path:
    return resolve_project_path(os.getenv("PANOCRAWLER_IMAGES_ROOT") or cfg.get("images_root", "images"))


def resolve_images_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == "images":
        path = Path(*path.parts[1:]) if len(path.parts) > 1 else Path()
    return get_images_root() / path
