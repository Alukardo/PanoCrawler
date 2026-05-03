import csv
import logging
import random
import time
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Tuple

import yaml
from dotenv import load_dotenv

from panorama import search_panoramas, get_panorama, PanoDownloadError, get_session, PanoramaSearchError
from panorama.process_images import detect_and_crop_black_edge

load_dotenv(Path(__file__).parent / ".apikey")

# ── Load config ───────────────────────────────────────────────────────────────
_CONFIG_PATH = Path(__file__).parent / "config.yaml"
with open(_CONFIG_PATH, encoding="utf-8") as f:
    _cfg = yaml.safe_load(f)

# ── 配置 ──────────────────────────────────────────────────────────────────────
images_dir = Path(__file__).parent / _cfg["images_dir"]
metadata_path = Path(__file__).parent / _cfg.get("metadata_path", "images/info.csv")
panoPath = images_dir
infoPath = metadata_path

# 每个全景下载后随机等待 3~8 秒，防止触发 Google 频率限制
MIN_DELAY = _cfg["min_delay"]
MAX_DELAY = _cfg["max_delay"]

locList: list[Tuple[float, float]] = [
    (25.045711097729114, 121.51134055812804),
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

FIELDNAMES = ["pano_id", "lat", "lon", "heading", "pitch", "roll", "date"]


# ── 初始化 info.csv（仅在文件不存在时写入表头）───────────────────────────────

def init_info() -> None:
    panoPath.mkdir(parents=True, exist_ok=True)
    infoPath.parent.mkdir(parents=True, exist_ok=True)
    if not infoPath.exists():
        with open(infoPath, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(FIELDNAMES)
        log.info("Created %s", infoPath)
    else:
        log.info("info.csv exists, will upsert records")


def write_info_records(records: dict[str, dict]) -> None:
    """Atomically persist panorama metadata to avoid partial-file corruption."""
    panoPath.mkdir(parents=True, exist_ok=True)
    infoPath.parent.mkdir(parents=True, exist_ok=True)

    temp_path: Path | None = None
    with NamedTemporaryFile(
        "w",
        encoding="utf-8",
        newline="",
        dir=infoPath.parent,
        prefix="info.",
        suffix=".tmp",
        delete=False,
    ) as temp_file:
        writer = csv.DictWriter(temp_file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(records.values())
        temp_path = Path(temp_file.name)

    try:
        temp_path.replace(infoPath)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


# ── 搜索并下载全景图 ─────────────────────────────────────────────────────────

def fetch_panoramas(loc: Tuple[float, float], isCurrent: bool) -> None:
    lat, lon = loc
    session = get_session()

    log.info("Searching near (%.6f, %.6f)...", lat, lon)
    try:
        panos = search_panoramas(lat=lat, lon=lon)
    except PanoramaSearchError as e:
        log.warning("Search failed near (%.6f, %.6f): %s", lat, lon, e)
        return
    log.info("Found %d panorama(s)", len(panos))

    if not panos:
        log.warning("No panoramas found for the given location.")
        return

    # ── 加载已有记录（全量读入，内存中 upsert）─────────────────────────────
    records: dict[str, dict] = {}
    if infoPath.exists():
        with open(infoPath, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                records[row["pano_id"]] = row

    # ── 遍历搜索结果，图片可用后再 upsert ───────────────────────────────────
    for i, pano in enumerate(panos):
        # isCurrent=True  → date 为空（当前最新全景）
        # isCurrent=False → date 非空（历史数据）
        if isCurrent and pano.date is not None:
            continue
        if not isCurrent and pano.date is None:
            continue

        row = {
            "pano_id": pano.pano_id,
            "lat":      pano.lat,
            "lon":      pano.lon,
            "heading":  pano.heading,
            "pitch":    pano.pitch,
            "roll":     pano.roll,
            "date":     pano.date,
        }
        was_recorded = pano.pano_id in records
        if not was_recorded:
            log.info(
                "  [%d/%d] + NEW  pano_id=%s  lat=%.6f  lon=%.6f  date=%s",
                i + 1, len(panos), pano.pano_id, pano.lat, pano.lon, pano.date,
            )
        else:
            log.info(
                "  [%d/%d] ~ UPDATE pano_id=%s",
                i + 1, len(panos), pano.pano_id,
            )

        # ── 图片下载（已有则跳过）───────────────────────────────────────────
        img_path = panoPath / f"{pano.pano_id}.png"
        if img_path.exists():
            log.info("  Already exists, skipping.")
            records[pano.pano_id] = row
        else:
            tmp_path = panoPath / f".{pano.pano_id}.download.tmp.png"
            processed_tmp_path = panoPath / f".{pano.pano_id}.processed.tmp.png"
            try:
                # zoom=3: 512×256 output, balanced quality vs file-size
                image = get_panorama(pano=pano, zoom=_cfg["zoom"], session=session)
                image.save(tmp_path, "png")

                # 黑边检测/裁剪（可选）
                if _cfg.get("process_on_download", False):
                    threshold = _cfg.get("black_threshold", 15)
                    cropped = detect_and_crop_black_edge(tmp_path, processed_tmp_path, threshold=threshold)
                    if cropped:
                        log.info("  Black edge cropped")
                    processed_tmp_path.replace(tmp_path)

                tmp_path.replace(img_path)

                log.info("  Saved: %s", img_path.name)
                records[pano.pano_id] = row
            except Exception as e:
                records.pop(pano.pano_id, None)
                log.warning("  [%s] skipped: %s", pano.pano_id, e)
                tmp_path.unlink(missing_ok=True)
                processed_tmp_path.unlink(missing_ok=True)

        # 节流：每个全景之间随机等待，避免频率限制
        if i < len(panos) - 1:
            delay = random.uniform(MIN_DELAY, MAX_DELAY)
            time.sleep(delay)

    # ── 全量写回 CSV（去重 + 更新后完整状态）─────────────────────────────────
    write_info_records(records)
    log.info("Wrote %d unique records to %s", len(records), infoPath)


# ── 入口 ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    init_info()
    for location in locList:
        fetch_panoramas(location, isCurrent=False)
