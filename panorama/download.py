import itertools
import logging
import time
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from io import BytesIO
from typing import Generator, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from PIL import Image
import yaml

from .search import Panorama

log = logging.getLogger(__name__)

# ── Load config ───────────────────────────────────────────────────────────────
_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
with open(_CONFIG_PATH, encoding="utf-8") as f:
    _cfg = yaml.safe_load(f)

# ── Constants ────────────────────────────────────────────────────────────────

DATE_THRESHOLD = datetime.strptime(_cfg["date_threshold"], "%Y-%m")
OUTPUT_SIZE = tuple(_cfg["output_size"])

# Tile server politely identifies itself
TILE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://maps.google.com/",
}


# ── Shared session ────────────────────────────────────────────────────────────

_session: requests.Session | None = None


def get_session() -> requests.Session:
    """
    Returns a re-useable session with connection pooling and retry logic.
    Automatically backs off on 403 / 429 / timeouts.
    """
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers.update(TILE_HEADERS)

        retry_strategy = Retry(
            total=3,
            backoff_factor=1.0,          # 1s, 2s, 4s  (urllib3 backoff)
            status_forcelist={403, 429, 500, 502, 503, 504},
            allowed_methods=["GET"],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=8,
            pool_maxsize=16,
        )
        _session.mount("https://cbk0.google.com", adapter)
        _session.mount("https://", adapter)
    return _session


# ── Dataclasses ──────────────────────────────────────────────────────────────

@dataclass
class TileInfo:
    x: int
    y: int
    fileurl: str


@dataclass
class Tile:
    x: int
    y: int
    image: Image.Image


# ── Tile helpers ─────────────────────────────────────────────────────────────

def get_width_and_height_from_zoom(zoom: int) -> Tuple[int, int]:
    return 2**zoom, 2 ** (zoom - 1)


def make_download_url(pano_id: str, zoom: int, x: int, y: int) -> str:
    return (
        "https://cbk0.google.com/cbk"
        f"?output=tile&panoid={pano_id}&zoom={zoom}&x={x}&y={y}"
    )


def fetch_panorama_tile(tile_info: TileInfo, session: requests.Session) -> Image.Image:
    """
    Downloads a single tile using the shared session.
    Retries up to 3 times with exponential backoff on transient errors.
    Raises on hard failures (403 without Retry-After, etc.).
    """
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            resp = session.get(tile_info.fileurl, stream=True, timeout=20)
            if resp.status_code == 403:
                # Hard block – no point retrying immediately
                raise PanoDownloadError(
                    f"403 Forbidden for {tile_info.fileurl}, "
                    "pano may not exist or is unavailable."
                )
            if resp.status_code == 429:
                # Rate-limited – wait and retry
                wait = int(resp.headers.get("Retry-After", 5))
                log.warning("429 rate-limited, waiting %ds before retry.", wait)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return Image.open(BytesIO(resp.content))
        except requests.RequestException:
            if attempt < max_attempts - 1:
                # Exponential backoff: 1s, 2s
                time.sleep(1.0 * (2 ** attempt) + random.uniform(0, 0.5))
            else:
                raise  # let caller handle


def iter_tile_info(pano_id: str, zoom: int) -> Generator[TileInfo, None, None]:
    width, height = get_width_and_height_from_zoom(zoom)
    for x, y in itertools.product(range(width), range(height)):
        yield TileInfo(
            x=x,
            y=y,
            fileurl=make_download_url(pano_id=pano_id, zoom=zoom, x=x, y=y),
        )


def iter_tiles(pano_id: str, zoom: int, session: requests.Session) -> Generator[Tile, None, None]:
    for info in iter_tile_info(pano_id, zoom):
        image = fetch_panorama_tile(info, session)
        yield Tile(x=info.x, y=info.y, image=image)


# ── Exceptions ───────────────────────────────────────────────────────────────

class PanoDownloadError(Exception):
    """Raised when a panorama tile fails to download."""
    pass


# ── Main download function ───────────────────────────────────────────────────

def get_panorama(pano: Panorama, zoom: int = 5, session: requests.Session | None = None) -> Image.Image:
    """
    Downloads and stitches a panorama.
    Uses a shared session for connection reuse and polite rate-limiting.
    Raises PanoDownloadError on hard failure.
    """
    if session is None:
        session = get_session()

    scale_width, scale_height = get_width_and_height_from_zoom(zoom)

    is_post_2017 = True
    if pano.date is not None:
        is_post_2017 = datetime.strptime(pano.date, "%Y-%m") > DATE_THRESHOLD

    real_width = 256 * (2**zoom) if is_post_2017 else 208 * 2**zoom
    real_height = 512 * (2**zoom) if is_post_2017 else 416 * (2**zoom)

    total_width = pano.scale[zoom][0][1]
    total_height = pano.scale[zoom][0][0]

    panorama = Image.new("RGB", (total_width, total_height))
    try:
        for tile in iter_tiles(pano_id=pano.pano_id, zoom=zoom, session=session):
            panorama.paste(im=tile.image, box=(tile.x * pano.tile[1], tile.y * pano.tile[0]))
            del tile
    except Exception as e:
        raise PanoDownloadError(f"Tile download failed for pano {pano.pano_id}: {e}") from e

    panorama = panorama.crop((0, 0, real_height, real_width))
    panorama = panorama.resize((total_width, total_height))
    clip_x = int(pano.heading / 360.0 * total_width)
    clip1 = panorama.crop((0, 0, clip_x, total_height))
    clip2 = panorama.crop((clip_x + 1, 0, total_width, total_height))

    panorama.paste(clip2, box=(0, 0))
    panorama.paste(clip1, box=(total_width - clip_x - 1, 0))
    panorama = panorama.resize(OUTPUT_SIZE)
    return panorama
