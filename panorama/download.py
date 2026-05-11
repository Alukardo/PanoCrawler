import itertools
import logging
import os
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

from .config import cfg as _cfg, resolve_images_path, resolve_project_path
from .search import Panorama
from .process_images import crop_black_edge_from_image
from .quality import apply_heading_adjustment
from .quota import GoogleAPIQuotaExceededError, record_failed_request, reserve_request

log = logging.getLogger(__name__)

# ── Load config ───────────────────────────────────────────────────────────────

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
            total=0,
            backoff_factor=1.0,
            status_forcelist=set(),
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


@dataclass
class PanoramaStages:
    raw: Image.Image
    base: Image.Image
    final: Image.Image
    zoom: int
    downloaded_tiles: int


# ── Tile API ─────────────────────────────────────────────────────────────────
def get_width_and_height_from_zoom(zoom: int) -> Tuple[int, int]:
    """
    Returns tile grid size (cols, rows) for a given zoom level.
    Maps Tile API: cols = 2**(zoom+1), rows = 2**zoom
    e.g. zoom=1 → 4×2 tiles (1024×512), zoom=2 → 8×4 tiles (2048×1024)
    """
    return 2 ** (zoom + 1), 2**zoom


def get_tile_grid_for_canvas(canvas_size: Tuple[int, int], tile_size: Tuple[int, int]) -> Tuple[int, int]:
    canvas_width, canvas_height = canvas_size
    tile_width, tile_height = tile_size
    if min(canvas_width, canvas_height, tile_width, tile_height) <= 0:
        raise ValueError("Canvas and tile dimensions must be positive")
    return (
        (canvas_width + tile_width - 1) // tile_width,
        (canvas_height + tile_height - 1) // tile_height,
    )


def choose_best_zoom_for_output(pano: Panorama, target_size: Tuple[int, int]) -> int:
    """
    Choose the smallest zoom level whose canvas meets or exceeds the target size.
    """
    if pano.zoom_resolutions:
        target_width, target_height = target_size
        for zoom, (width, height) in enumerate(pano.zoom_resolutions):
            if width >= target_width and height >= target_height:
                return zoom
        return len(pano.zoom_resolutions) - 1
    return _cfg.get("zoom", 1)


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
        try:
            reserve_request("session_requests")
        except GoogleAPIQuotaExceededError as e:
            raise PanoDownloadError(str(e)) from e
        try:
            resp = requests.post(
                self.SESSION_URL,
                params={"key": self.api_key},
                json={"mapType": "streetview", "language": "en-US", "region": "US"},
                headers={"Content-Type": "application/json"},
                timeout=TILE_TIMEOUT,
            )
            resp.raise_for_status()
            token = resp.json()["session"]
        except requests.RequestException as e:
            record_failed_request("session_failures")
            raise PanoDownloadError(f"Failed to create tile session: {e}") from e
        except (KeyError, TypeError, ValueError) as e:
            record_failed_request("session_failures")
            raise PanoDownloadError("Failed to create tile session: missing session token") from e
        if not token:
            record_failed_request("session_failures")
            raise PanoDownloadError("Failed to create tile session: missing session token")
        return token

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
                reserve_request("tile_requests")
            except GoogleAPIQuotaExceededError as e:
                raise PanoDownloadError(str(e)) from e
            try:
                resp = session.get(url, stream=True, timeout=TILE_TIMEOUT)
                if resp.status_code == 400:
                    record_failed_request("tile_400_out_of_range")
                    raise PanoTileOutOfRangeError(f"400 Bad Request for tile ({x},{y}) — out of range for this pano")
                if resp.status_code == 403:
                    record_failed_request("tile_403_forbidden")
                    raise PanoDownloadError(f"403 Forbidden — pano may not exist: {pano_id}")
                if resp.status_code == 429:
                    record_failed_request("tile_429_rate_limited")
                    wait = int(resp.headers.get("Retry-After", 5))
                    log.warning("429 rate-limited, waiting %ds", wait)
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                return Image.open(BytesIO(resp.content))
            except requests.RequestException as e:
                record_failed_request("tile_failures")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(1.0 * (2**attempt) + random.uniform(0, 0.5))
                else:
                    raise PanoDownloadError(f"Failed to download tile ({x},{y}): {e}") from e
        raise PanoDownloadError(f"Failed to download tile ({x},{y}) after {MAX_RETRIES} attempts")

    def iter_tiles(
        self,
        pano_id: str,
        zoom: int,
        session: requests.Session,
        cols: int | None = None,
        rows: int | None = None,
    ) -> Generator[Tile, None, None]:
        """Iterate all tiles. Skips tiles that are out of range (400) for this pano."""
        if cols is None or rows is None:
            cols, rows = get_width_and_height_from_zoom(zoom)
        for x, y in itertools.product(range(cols), range(rows)):
            try:
                image = self.download_tile(pano_id, zoom, x, y, session)
                yield Tile(x=x, y=y, image=image)
            except PanoTileOutOfRangeError as e:
                log.debug("Skipping tile (%d,%d): %s", x, y, e)
                continue


# ── Factory ──────────────────────────────────────────────────────────────────
def get_downloader(api_key: str | None = None) -> MapsTileAPIDownloader:
    if api_key is None:
        api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise PanoDownloadError("api_key must be set via the GOOGLE_API_KEY environment variable")
    return MapsTileAPIDownloader(api_key=api_key)


# ── Exceptions ───────────────────────────────────────────────────────────────
class PanoDownloadError(Exception):
    pass


class PanoTileOutOfRangeError(PanoDownloadError):
    pass


# ── Main download functions ──────────────────────────────────────────────────

def get_panorama(
    pano: Panorama,
    zoom: int | None = None,
    session: requests.Session | None = None,
) -> Image.Image:
    """
    Downloads and stitches a panorama.

    Pipeline: 拼图 → 黑边裁剪 → resize(OUTPUT_SIZE) → heading旋转
    """
    return get_panorama_stages(pano=pano, zoom=zoom, session=session).final


def get_panorama_stages(
    pano: Panorama,
    zoom: int | None = None,
    session: requests.Session | None = None,
) -> PanoramaStages:
    if session is None:
        session = get_session()
    if zoom is None:
        zoom = choose_best_zoom_for_output(pano, OUTPUT_SIZE)
    total_width, total_height = pano.get_canvas_size(zoom)
    downloader = get_downloader()
    raw = Image.new("RGB", (total_width, total_height))
    downloaded = 0
    try:
        tile_width, tile_height = (pano.tile[1], pano.tile[0]) if pano.tile and len(pano.tile) >= 2 else (512, 512)
        cols, rows = get_tile_grid_for_canvas((total_width, total_height), (tile_width, tile_height))
        for tile in downloader.iter_tiles(pano_id=pano.pano_id, zoom=zoom, session=session, cols=cols, rows=rows):
            raw.paste(im=tile.image, box=(tile.x * tile_width, tile.y * tile_height))
            downloaded += 1
    except PanoDownloadError:
        raise
    except Exception as e:
        raise PanoDownloadError(f"Tile download failed for pano {pano.pano_id}: {e}") from e
    if downloaded == 0:
        raise PanoDownloadError(f"No tiles downloaded for pano {pano.pano_id}")
    cropped = crop_black_edge_from_image(raw, threshold=_cfg.get("black_threshold", 15))
    base = cropped.resize(OUTPUT_SIZE)
    final = apply_heading_adjustment(base, pano.heading)
    return PanoramaStages(raw=raw, base=base, final=final, zoom=zoom, downloaded_tiles=downloaded)


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
        output_dir = resolve_images_path(_cfg.get("output_dir", _cfg.get("images_dir", "pano")))
    else:
        output_dir = resolve_project_path(output_dir)

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
