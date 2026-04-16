import itertools
import time
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from typing import Generator, Tuple

import requests
from PIL import Image

from .search import Panorama

# Panorama date threshold: differentiate old vs new format
DATE_THRESHOLD = datetime.strptime("2017-09", "%Y-%m")
OUTPUT_SIZE = (1024, 512)


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


def get_width_and_height_from_zoom(zoom: int) -> Tuple[int, int]:
    """
    Returns the width and height of a panorama at a given zoom level, depends on the
    zoom level.
    """
    return 2**zoom, 2 ** (zoom - 1)


def make_download_url(pano_id: str, zoom: int, x: int, y: int) -> str:
    """
    Returns the URL to download a tile.
    """
    return (
        "https://cbk0.google.com/cbk"
        f"?output=tile&panoid={pano_id}&zoom={zoom}&x={x}&y={y}"
    )


def fetch_panorama_tile(tile_info: TileInfo) -> Image.Image:
    """
    Tries to download a tile, returns a PIL Image. Retries on connection error.
    """
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = requests.get(tile_info.fileurl, stream=True, timeout=10)
            response.raise_for_status()
            return Image.open(BytesIO(response.content))
        except requests.RequestException:
            if attempt < max_retries - 1:
                print(f"Connection error (attempt {attempt + 1}/{max_retries}). Retrying in 1 seconds.")
                time.sleep(1)
            else:
                raise


def iter_tile_info(pano_id: str, zoom: int) -> Generator[TileInfo, None, None]:
    """
    Generate a list of a panorama's tiles and their position.
    """
    width, height = get_width_and_height_from_zoom(zoom)
    for x, y in itertools.product(range(width), range(height)):
        yield TileInfo(
            x=x,
            y=y,
            fileurl=make_download_url(pano_id=pano_id, zoom=zoom, x=x, y=y),
        )


def iter_tiles(pano_id: str, zoom: int) -> Generator[Tile, None, None]:
    for info in iter_tile_info(pano_id, zoom):
        image = fetch_panorama_tile(info)
        yield Tile(x=info.x, y=info.y, image=image)


def get_panorama(pano: Panorama, zoom: int = 5) -> Image.Image:
    """
    Downloads and stitches a panorama.
    """
    scale_width, scale_height = get_width_and_height_from_zoom(zoom)

    time_state = True
    if pano.date is not None:
        time_state = datetime.strptime(pano.date, "%Y-%m") > DATE_THRESHOLD

    real_width = 256 * (2**zoom) if time_state else 208 * 2**zoom
    real_height = 512 * (2**zoom) if time_state else 416 * (2**zoom)

    total_width = pano.scale[zoom][0][1]
    total_height = pano.scale[zoom][0][0]

    panorama = Image.new("RGB", (total_width, total_height))
    for tile in iter_tiles(pano_id=pano.pano_id, zoom=zoom):
        panorama.paste(im=tile.image, box=(tile.x * pano.tile[1], tile.y * pano.tile[0]))
        del tile
    panorama = panorama.crop((0, 0, real_height, real_width))
    panorama = panorama.resize((total_width, total_height))
    clip_x = int(pano.heading / 360.0 * total_width)
    clip1 = panorama.crop((0, 0, clip_x, total_height))
    clip2 = panorama.crop((clip_x + 1, 0, total_width, total_height))

    panorama.paste(clip2, box=(0, 0))
    panorama.paste(clip1, box=(total_width - clip_x - 1, 0))
    panorama = panorama.resize(OUTPUT_SIZE)
    return panorama
