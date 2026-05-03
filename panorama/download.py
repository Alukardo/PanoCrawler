import itertools
import logging
import time
import random
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Generator, Tuple

import requests
from PIL import Image
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import yaml

from .search import Panorama
from .process_images import crop_black_edge_from_image

log = logging.getLogger(__name__)

# ── Load config ───────────────────────────────────────────────────────────────
_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
with open(_CONFIG_PATH, encoding="utf-8") as f:
    _cfg = yaml.safe_load(f)

# ── Constants ───────────────────────────────────────────────────────────────
OUTPUT_SIZE = tuple(_cfg["output_size"])
TILE_TIMEOUT = _cfg.get("tile_timeout", 20)
MAX_RETRIES = _cfg.get("max_retries", 3)

_TILE_HEADERS = {
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
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers.update(_TILE_HEADERS)
        retry_strategy = Retry(
            total=3,
            backoff_factor=1.0,
            status_forcelist={403, 429, 500, 502, 503, 504},
            allowed_methods=["GET"],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=8,
            pool_maxsize=16,
        )
        _session.mount("https://", adapter)
    return _session


# ── Dataclasses ──────────────────────────────────────────────────────────────
@dataclass
class Tile:
    x: int
    y: int
    image: Image.Image


# ── Tile API ─────────────────────────────────────────────────────────────────
def get_width_and_height_from_zoom(zoom: int) -> Tuple[int, int]:
    """
    Returns tile grid size (cols, rows) for a given zoom level.
    Maps Tile API: cols = 2**(zoom+1), rows = 2**zoom
    e.g. zoom=1 → 4×2 tiles (1024×512), zoom=2 → 8×4 tiles (2048×1024)
    """
    return 2 ** (zoom + 1), 2**zoom


# ── Official Maps Tile API Downloader ────────────────────────────────────────
class MapsTileAPIDownloader:
    """
    Official Google Maps Tile API downloader.

    1. POST https://tile.googleapis.com/v1/createSession  (mapType=streetview)
    2. Use returned session token to download tiles via:
       https://tile.googleapis.com/v1/streetview/tiles/{zoom}/{x}/{y}
           ?session=<token>&key=<api_key>&panoId=<pano_id>

    Each tile is 512×512 pixels.
    """

    SESSION_URL = "https://tile.googleapis.com/v1/createSession"
    TILE_URL_TEMPLATE = (
        "https://tile.googleapis.com/v1/streetview/tiles/{zoom}/{x}/{y}"
        "?session={session}&key={api_key}&panoId={pano_id}"
    )

    def __init__(self, api_key: str):
        self.api_key = api_key
        self._session_cache: dict[str, str] = {}

    def _create_session(self) -> str:
        resp = requests.post(
            self.SESSION_URL,
            params={"key": self.api_key},
            json={"mapType": "streetview", "language": "en-US", "region": "US"},
            headers={"Content-Type": "application/json"},
            timeout=TILE_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()["session"]

    def download_tile(
        self,
        pano_id: str,
        zoom: int,
        x: int,
        y: int,
        session: requests.Session,
    ) -> Image.Image:
        if pano_id not in self._session_cache:
            self._session_cache[pano_id] = self._create_session()
        token = self._session_cache[pano_id]

        url = self.TILE_URL_TEMPLATE.format(
            zoom=zoom, x=x, y=y, session=token, api_key=self.api_key, pano_id=pano_id
        )
        for attempt in range(MAX_RETRIES):
            try:
                resp = session.get(url, stream=True, timeout=TILE_TIMEOUT)
                if resp.status_code == 400:
                    raise PanoDownloadError(f"400 Bad Request for tile ({x},{y}) — out of range for this pano")
                if resp.status_code == 403:
                    raise PanoDownloadError(f"403 Forbidden — pano may not exist: {pano_id}")
                if resp.status_code == 429:
                    wait = int(resp.headers.get("Retry-After", 5))
                    log.warning("429 rate-limited, waiting %ds", wait)
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                return Image.open(BytesIO(resp.content))
            except requests.RequestException:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(1.0 * (2**attempt) + random.uniform(0, 0.5))
                else:
                    raise
        raise PanoDownloadError(f"Failed to download tile ({x},{y}) after {MAX_RETRIES} attempts")

    def iter_tiles(
        self,
        pano_id: str,
        zoom: int,
        session: requests.Session,
    ) -> Generator[Tile, None, None]:
        """Iterate all tiles. Skips tiles that are out of range (400) for this pano."""
        cols, rows = get_width_and_height_from_zoom(zoom)
        for x, y in itertools.product(range(cols), range(rows)):
            try:
                image = self.download_tile(pano_id, zoom, x, y, session)
                yield Tile(x=x, y=y, image=image)
            except PanoDownloadError as e:
                log.debug("Skipping tile (%d,%d): %s", x, y, e)
                continue


# ── Factory ──────────────────────────────────────────────────────────────────
def get_downloader(api_key: str | None = None) -> MapsTileAPIDownloader:
    if api_key is None:
        api_key = _cfg.get("api_key", "")
    if not api_key:
        raise PanoDownloadError("api_key must be set")
    return MapsTileAPIDownloader(api_key=api_key)


# ── Exceptions ───────────────────────────────────────────────────────────────
class PanoDownloadError(Exception):
    pass


# ── Main download functions ──────────────────────────────────────────────────

def get_panorama(
    pano: Panorama,
    zoom: int = 5,
    session: requests.Session | None = None,
) -> Image.Image:
    """
    Downloads and stitches a panorama.

    Pipeline: 拼图 → 黑边裁剪 → resize(OUTPUT_SIZE) → heading旋转
    """
    if session is None:
        session = get_session()

    total_width = pano.scale[zoom][0][1]
    total_height = pano.scale[zoom][0][0]

    downloader = get_downloader(api_key=_cfg.get("api_key", ""))

    # 1. 拼图
    panorama = Image.new("RGB", (total_width, total_height))
    downloaded = 0
    try:
        for tile in downloader.iter_tiles(pano_id=pano.pano_id, zoom=zoom, session=session):
            panorama.paste(im=tile.image, box=(tile.x * pano.tile[1], tile.y * pano.tile[0]))
            downloaded += 1
    except Exception as e:
        raise PanoDownloadError(f"Tile download failed for pano {pano.pano_id}: {e}") from e
    if downloaded == 0:
        raise PanoDownloadError(f"No tiles downloaded for pano {pano.pano_id}")

    # 2. 黑边裁剪（纯像素，无损）
    panorama = crop_black_edge_from_image(panorama, threshold=_cfg.get("black_threshold", 15))

    # 3. resize 到最终尺寸
    panorama = panorama.resize(OUTPUT_SIZE)

    # 4. heading 旋转（像素剪切/粘贴，无损）
    heading_norm = pano.heading % 360
    clip_x = int(heading_norm / 360.0 * OUTPUT_SIZE[0])
    if clip_x > 0:
        clip1 = panorama.crop((0, 0, clip_x, OUTPUT_SIZE[1]))
        clip2 = panorama.crop((clip_x, 0, OUTPUT_SIZE[0], OUTPUT_SIZE[1]))
        panorama.paste(clip2, box=(0, 0))
        panorama.paste(clip1, box=(OUTPUT_SIZE[0] - clip_x, 0))

    return panorama


def download_by_pano_id(
    pano_id: str,
    zoom: int = 1,
    output_dir: str | Path | None = None,
) -> Image.Image:
    """
    Downloads a panorama by panoID without full Panorama metadata.
    Infers canvas size from zoom level. Tiles are 512×512 each.
    """
    if output_dir is None:
        output_dir = _cfg["images_dir"]
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)

    session = get_session()
    downloader = get_downloader()

    cols, rows = get_width_and_height_from_zoom(zoom)
    tile_size = 512
    canvas = Image.new("RGB", (cols * tile_size, rows * tile_size))

    downloaded = 0
    for tile in downloader.iter_tiles(pano_id=pano_id, zoom=zoom, session=session):
        canvas.paste(tile.image, (tile.x * tile_size, tile.y * tile_size))
        downloaded += 1

    if downloaded == 0:
        raise PanoDownloadError(f"No tiles downloaded for pano {pano_id}")

    out_path = output_dir / f"{pano_id}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, "PNG")
    log.info("Saved panorama %s (%dx%d, %d tiles): %s", pano_id, canvas.width, canvas.height, downloaded, out_path)

    return canvas
