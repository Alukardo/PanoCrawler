import csv
import logging
import random
import time
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Tuple

from panorama import search_panoramas, get_panorama, PanoDownloadError, get_session, PanoramaSearchError
from panorama.config import cfg as _cfg, resolve_images_path
from panorama.quota import GoogleAPIQuotaExceededError

# ── Load config ───────────────────────────────────────────────────────────────

# ── 配置 ──────────────────────────────────────────────────────────────────────
images_dir = resolve_images_path(_cfg.get("images_dir", "pano"))
metadata_path = resolve_images_path(_cfg.get("metadata_path", "info.csv"))
panoPath = images_dir
infoPath = metadata_path

# 每个全景下载后随机等待 3~8 秒，防止触发 Google 频率限制
MIN_DELAY = _cfg["min_delay"]
MAX_DELAY = _cfg["max_delay"]

locList: list[Tuple[float, float]] = [
    (25.045711097729114, 121.51134055812804),
]

DEFAULT_RANDOM_REGIONS: list[tuple[float, float, float, float]] = [
    (25.0, 49.0, -124.0, -67.0),
    (7.0, 22.0, -100.0, -75.0),
    (-34.0, 5.0, -76.0, -35.0),
    (36.0, 60.0, -10.0, 30.0),
    (50.0, 58.0, -8.0, 2.0),
    (22.0, 46.0, 120.0, 146.0),
    (-8.0, 22.0, 95.0, 122.0),
    (-45.0, -10.0, 112.0, 178.0),
    (8.0, 30.0, 68.0, 90.0),
    (-35.0, -22.0, 16.0, 33.0),
]
RANDOM_CRAWL_TARGET_NEW = int(_cfg.get("random_crawl_target_new", 200))
RANDOM_CRAWL_MAX_SEARCHES = int(_cfg.get("random_crawl_max_searches", 2000))
CRAWL_MODE = _cfg.get("crawl_mode", "random_incremental")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

FIELDNAMES = _cfg.get("fieldnames", ["pano_id", "lat", "lon", "heading", "pitch", "roll", "date", "search_point"])


def is_quota_error(exc: BaseException) -> bool:
    current: BaseException | None = exc
    while current is not None:
        if isinstance(current, GoogleAPIQuotaExceededError):
            return True
        current = current.__cause__ or current.__context__
    return False


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


def load_info_records() -> dict[str, dict]:
    records: dict[str, dict] = {}
    if infoPath.exists():
        with open(infoPath, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("pano_id"):
                    records[row["pano_id"]] = row
    return records


def format_search_point(search_point: Tuple[float, float] | None) -> str:
    if search_point is None:
        return ""
    lat, lon = search_point
    return f"[{lat:.6f}, {lon:.6f}]"


def build_info_row(pano, search_point: Tuple[float, float] | None = None) -> dict:
    return {
        "pano_id": pano.pano_id,
        "lat": pano.lat,
        "lon": pano.lon,
        "heading": pano.heading,
        "pitch": pano.pitch,
        "roll": pano.roll,
        "date": pano.date,
        "search_point": format_search_point(search_point),
    }


def random_location(regions: list[tuple[float, float, float, float]] | None = None) -> Tuple[float, float]:
    lat_min, lat_max, lon_min, lon_max = random.choice(regions or DEFAULT_RANDOM_REGIONS)
    return random.uniform(lat_min, lat_max), random.uniform(lon_min, lon_max)


def download_missing_panorama(
    pano,
    records: dict[str, dict],
    session,
    search_point: Tuple[float, float] | None = None,
) -> bool:
    row = build_info_row(pano, search_point)
    panoPath.mkdir(parents=True, exist_ok=True)
    img_path = panoPath / f"{pano.pano_id}.png"
    if img_path.exists():
        log.info("  Download skipped: pano_id=%s existing_path=%s", pano.pano_id, img_path)
        records[pano.pano_id] = row
        return False

    tmp_path = panoPath / f".{pano.pano_id}.download.tmp.png"
    try:
        log.info("  Download start: pano_id=%s target=%s", pano.pano_id, img_path)
        image = get_panorama(pano=pano, session=session)
        image.save(tmp_path, "png")
        tmp_size = tmp_path.stat().st_size
        if tmp_size <= 0:
            raise PanoDownloadError(f"Downloaded empty image for pano {pano.pano_id}")
        if _cfg.get("process_on_download", False):
            log.info("  Black edge already handled by get_panorama")
        tmp_path.replace(img_path)
        saved_size = img_path.stat().st_size
        log.info("  Download saved: pano_id=%s bytes=%d path=%s", pano.pano_id, saved_size, img_path)
        records[pano.pano_id] = row
        return True
    except Exception as e:
        log.warning("  Download failed: pano_id=%s error=%s", pano.pano_id, e)
        records.pop(pano.pano_id, None)
        tmp_path.unlink(missing_ok=True)
        raise


# ── 搜索并下载全景图 ─────────────────────────────────────────────────────────

def fetch_panoramas(loc: Tuple[float, float], isCurrent: bool) -> int:
    lat, lon = loc
    session = get_session()

    log.info("Searching near (%.6f, %.6f)...", lat, lon)
    try:
        panos = search_panoramas(lat=lat, lon=lon)
    except PanoramaSearchError as e:
        log.warning("Search failed near (%.6f, %.6f): %s", lat, lon, e)
        return 0
    log.info("Found %d panorama(s)", len(panos))

    if not panos:
        log.warning("No panoramas found for the given location.")
        return 0

    # ── 加载已有记录（全量读入，内存中 upsert）─────────────────────────────
    records = load_info_records()
    added = 0

    # ── 遍历搜索结果，图片可用后再 upsert ───────────────────────────────────
    for i, pano in enumerate(panos):
        # isCurrent=True  → date 为空（当前最新全景）
        # isCurrent=False → date 非空（历史数据）
        if isCurrent and pano.date is not None:
            continue
        if not isCurrent and pano.date is None:
            continue

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
            records[pano.pano_id] = build_info_row(pano, (lat, lon))
        else:
            try:
                downloaded = download_missing_panorama(pano, records, session, (lat, lon))
                log.info("  Saved: %s", img_path.name)
                if downloaded:
                    added += 1
            except Exception as e:
                if is_quota_error(e):
                    log.warning("  Google API daily soft limit reached, stopping crawl: %s", e)
                    break
                log.warning("  [%s] skipped: %s", pano.pano_id, e)

        # 节流：每个全景之间随机等待，避免频率限制
        if i < len(panos) - 1:
            delay = random.uniform(MIN_DELAY, MAX_DELAY)
            time.sleep(delay)

    # ── 全量写回 CSV（去重 + 更新后完整状态）─────────────────────────────────
    write_info_records(records)
    log.info("Wrote %d unique records to %s", len(records), infoPath)
    return added


def fetch_random_incremental_panoramas(
    target_new: int = RANDOM_CRAWL_TARGET_NEW,
    max_searches: int = RANDOM_CRAWL_MAX_SEARCHES,
) -> int:
    session = get_session()
    records = load_info_records()
    added = 0
    searches = 0

    while added < target_new and searches < max_searches:
        lat, lon = random_location()
        searches += 1
        log.info("Random search %d/%d near (%.6f, %.6f)", searches, max_searches, lat, lon)
        try:
            panos = search_panoramas(lat=lat, lon=lon)
        except PanoramaSearchError as e:
            if is_quota_error(e):
                log.warning("Google API daily soft limit reached, stopping crawl: %s", e)
                break
            log.warning("Search failed near (%.6f, %.6f): %s", lat, lon, e)
            continue

        random.shuffle(panos)
        for pano in panos:
            if added >= target_new:
                break
            if pano.pano_id in records and (panoPath / f"{pano.pano_id}.png").exists():
                continue
            try:
                if download_missing_panorama(pano, records, session, (lat, lon)):
                    added += 1
                    write_info_records(records)
                    log.info("Random crawl downloaded and added %d/%d: %s", added, target_new, pano.pano_id)
            except Exception as e:
                if is_quota_error(e):
                    write_info_records(records)
                    log.warning("Google API daily soft limit reached, stopping crawl: %s", e)
                    return added
                log.warning("  [%s] skipped: %s", pano.pano_id, e)
            if added < target_new:
                delay = random.uniform(MIN_DELAY, MAX_DELAY)
                time.sleep(delay)

    write_info_records(records)
    log.info("Random crawl added %d new panoramas after %d search(es)", added, searches)
    return added


# ── 入口 ─────────────────────────────────────────────────────────────────────

def main() -> int:
    init_info()
    if CRAWL_MODE == "random_incremental":
        fetch_random_incremental_panoramas()
    else:
        for location in locList:
            fetch_panoramas(location, isCurrent=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
