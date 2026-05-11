import json
import re
from typing import List, Optional

import requests
from pydantic import BaseModel
from requests.models import Response

from .config import cfg as _cfg
from .quota import GoogleAPIQuotaExceededError, record_failed_request, reserve_request

SEARCH_TIMEOUT = _cfg.get("search_timeout", 15)
SEARCH_URL_BASE = _cfg.get("search_url_base", "https://maps.googleapis.com/maps/api/js/GeoPhotoService.SingleImageSearch")


class PanoramaSearchError(Exception):
    pass


class Panorama(BaseModel):
    pano_id: str
    lat: float
    lon: float
    heading: float
    pitch: Optional[float] = None
    roll: Optional[float] = None
    date: Optional[str] = None
    scale: Optional[list] = None
    zoom_resolutions: Optional[list[tuple[int, int]]] = None
    tile: Optional[list] = None


    def print_info(self) -> None:
        print("pano_id : " + self.pano_id)
        print("loc: " + "(" + str(self.lat) + " , " + str(self.lon) + ")")
        print("heading : " + str(self.heading))
        print("pitch   : " + str(self.pitch))
        print("roll    : " + str(self.roll))
        print("date    : " + str(self.date))
        print("scale   : " + str(self.scale))
        print("zoom_resolutions : " + str(self.zoom_resolutions))


    def save_to_file(self, file) -> None:
        """Write panorama data to a file handle or csv.writer."""
        row = [self.pano_id, self.lat, self.lon, self.heading,
               self.pitch, self.roll, self.date, ""]
        try:
            file.writerow(row)
        except AttributeError:
            # fallback: plain file handle
            ps = ','.join(str(v) for v in row) + '\n'
            file.writelines(ps)

    def get_canvas_size(self, zoom: int) -> tuple[int, int]:
        """
        Return the panorama canvas size for a given zoom level.

        `scale` is encoded as a list of entries where each zoom value maps to a
        [height, width] pair:

            scale[zoom][0] == [height, width]
        """
        if self.scale is None:
            raise ValueError("Panorama scale metadata is missing")
        try:
            height, width = self.scale[zoom][0]
        except (IndexError, TypeError) as e:
            raise ValueError(f"Invalid zoom level {zoom} for this panorama") from e
        return width, height

    def get_zoom_resolutions(self) -> list[tuple[int, int]]:
        """
        Return available zoom resolutions as a list of (width, height) tuples.
        """
        if self.scale is None:
            raise ValueError("Panorama scale metadata is missing")
        try:
            return [(entry[0][1], entry[0][0]) for entry in self.scale]
        except (TypeError, IndexError) as e:
            raise ValueError("Invalid scale metadata format") from e


def make_search_url(lat: float, lon: float) -> str:
    """
    Builds the URL of the script on Google's servers that returns the closest
    panorama (ids) to a give GPS coordinate.
    """
    url = (
        SEARCH_URL_BASE
        + "?pb=!1m5!1sapiv3!5sUS!11m2!1m1!1b0!2m4!1m2!3d{0:}!4d{1:}!2d50!3m10"
        "!2m2!1sen!2sGB!9m1!1e2!11m4!1m3!1e2!2b1!3e2!4m10!1e1!1e2!1e3!1e4"
        "!1e8!1e6!5m1!1e2!6m1!1e2"
        "&callback=callbackfunc"
    )
    return url.format(lat, lon)


def search_request(lat: float, lon: float) -> Response:
    """
    Gets the response of the script on Google's servers that returns the
    closest panorama (ids) to a give GPS coordinate.
    """
    url = make_search_url(lat, lon)
    try:
        reserve_request("search_requests")
    except GoogleAPIQuotaExceededError as e:
        raise PanoramaSearchError(str(e)) from e
    try:
        resp = requests.get(url, timeout=SEARCH_TIMEOUT)
        resp.raise_for_status()
        return resp
    except requests.RequestException as e:
        record_failed_request("search_failures")
        raise PanoramaSearchError(f"Search request failed for ({lat}, {lon}): {e}") from e


def extract_panoramas(text: str) -> List[Panorama]:
    """
    Given a valid response from the panoids endpoint, return a list of all the
    panoids.
    """

    # The response is actually JavaScript code. It's a function with a single
    # input which is a huge deeply nested array of items.
    match = re.search(r"callbackfunc\(\s*(.*)\s*\)$", text.strip())
    if match is None:
        raise PanoramaSearchError("Search response did not contain callbackfunc payload")

    try:
        data = json.loads(match.group(1))

        if data == [[5, "generic", "Search returned no images."]]:
            return []

        subset = data[1][5][0]
        scale = data[1][2][3][0]
        raw_panos = subset[3][0]

        tile_size = data[1][2][3][1]
        zoom_resolutions = [(entry[0][1], entry[0][0]) for entry in scale]

        if len(subset) < 9 or subset[8] is None:
            raw_dates = []
        else:
            raw_dates = subset[8]

        # Build date lookup by pano_id (robust — order/slice doesn't matter)
        date_map = {
            raw_panos[d[0]][0][1]: f"{d[1][0]}-{d[1][1]:02d}"
            for d in raw_dates
        }

        return [
            Panorama(
                pano_id=pano[0][1],
                lat=pano[2][0][2],
                lon=pano[2][0][3],
                heading=pano[2][2][0],
                pitch=pano[2][2][1] if len(pano[2][2]) >= 2 else None,
                roll=pano[2][2][2] if len(pano[2][2]) >= 3 else None,
                date=date_map.get(pano[0][1]),
                scale=scale,
                zoom_resolutions=zoom_resolutions,
                tile=tile_size
            )
            for pano in raw_panos
        ]
    except (json.JSONDecodeError, IndexError, KeyError, TypeError, ValueError) as e:
        raise PanoramaSearchError("Could not parse panorama search response") from e


def search_panoramas(lat: float, lon: float) -> List[Panorama]:
    """
    Gets the closest panorama (ids) to the GPS coordinates.
    """

    resp = search_request(lat, lon)
    pans = extract_panoramas(resp.text)
    return pans
