import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from panorama import search_panoramas
from panorama import get_location_meta
from panorama import get_streetview
from panorama import get_panorama
from panorama import get_panorama_meta
from process import clean
import csv

GoogleAPIKey = os.environ.get("GOOGLE_API_KEY", "")
panoId = "z80QZ1_QgCbYwj7RrmlS0Q"
panoPath = "../images/pano/"
infoPath = os.path.join(panoPath, "info.csv")

loc0 = {"lat": 25.017331619756757, "lon": 121.53579493917834}
loc1 = {"lat": 41.898220819756757, "lon": 12.47648043917834}

loc01 = {"lat": 25.038147776102537, "lon": 121.56977877482196}
loc02 = {"lat": 25.014466835352888, "lon": 121.54363900896685}

locList = [(25.045711097729114, 121.51134055812804)]


def init_info() -> None:
    clean(panoPath)
    with open(infoPath, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pano_id", "lat", "lon", "heading", "pitch", "roll", "date"])


def print_pano(loc: tuple[float, float], isCurrent: bool) -> None:
    panos = search_panoramas(lat=loc[0], lon=loc[1])
    print(f"#Panos Count: {len(panos)}")
    with open(infoPath, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for pano in panos:
            if isCurrent and pano.date is not None:
                continue
            if not isCurrent and pano.date is None:
                continue
            print("##########################################")
            pano.print()
            pano.saveFile(writer)
            image = get_panorama(pano=pano, zoom=3)
            image.save(os.path.join(panoPath, f"{pano.pano_id}.png"), "png")
            print(f"panoImg : {panoPath}{pano.pano_id}.png")
            print("------------------------------------------")


if __name__ == "__main__":
    init_info()
    for location in locList:
        print_pano(location, isCurrent=False)
