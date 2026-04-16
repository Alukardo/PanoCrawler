import os
import csv
import logging
from pathlib import Path

from dotenv import load_dotenv
from panorama import search_panoramas
from panorama import get_panorama, PanoDownloadError

load_dotenv(Path(__file__).parent / ".env")

# ── 配置 ──────────────────────────────────────────────────────────────────────
panoPath = Path(__file__).parent.parent / "images" / "pano"
infoPath = panoPath / "info.csv"

# 搜索坐标列表：(纬度, 经度)
locList = [
    (25.045711097729114, 121.51134055812804),
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── 初始化 info.csv（仅在文件不存在时写入表头）───────────────────────────────

def init_info() -> None:
    panoPath.mkdir(parents=True, exist_ok=True)
    if not infoPath.exists():
        with open(infoPath, "a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["pano_id", "lat", "lon", "heading", "pitch", "roll", "date"])
        log.info("Created %s", infoPath)
    else:
        log.info("info.csv exists, appending data only")


# ── 搜索并下载全景图 ─────────────────────────────────────────────────────────

def fetch_panoramas(loc: tuple[float, float], isCurrent: bool) -> None:
    lat, lon = loc
    log.info("Searching near (%.6f, %.6f)...", lat, lon)
    panos = search_panoramas(lat=lat, lon=lon)
    log.info("Found %d panorama(s)", len(panos))

    if not panos:
        log.warning("No panoramas found for the given location.")
        return

    with open(infoPath, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for pano in panos:
            # isCurrent=True  → date 为空（当前最新）
            # isCurrent=False → date 非空（历史数据）
            if isCurrent and pano.date is not None:
                continue
            if not isCurrent and pano.date is None:
                continue
            row = [
                pano.pano_id,
                pano.lat,
                pano.lon,
                pano.heading,
                pano.pitch,
                pano.roll,
                pano.date,
            ]
            writer.writerow(row)
            log.info(
                "  pano_id=%s  lat=%.6f  lon=%.6f  date=%s",
                pano.pano_id, pano.lat, pano.lon, pano.date,
            )

            # 下载全景图（避免重复下载）
            img_path = panoPath / f"{pano.pano_id}.png"
            if img_path.exists():
                log.info("  Already exists, skipping: %s", img_path.name)
            else:
                try:
                    image = get_panorama(pano=pano, zoom=3)
                    image.save(img_path, "png")
                    log.info("  Saved: %s", img_path.name)
                except PanoDownloadError as e:
                    log.warning("  [%s] skipped: %s", pano.pano_id, e)


# ── 入口 ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    init_info()
    for location in locList:
        fetch_panoramas(location, isCurrent=False)
