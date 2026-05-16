import csv
import logging
import math
import random
import time
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Iterable, Tuple

from panorama import search_panoramas, get_panorama, PanoDownloadError, get_session, PanoramaSearchError
from panorama.config import cfg as _cfg, resolve_images_path
from panorama.quota import GoogleAPIQuotaExceededError
from panorama.search import Panorama

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
SKIP_LOW_PANO_SEARCH = bool(_cfg.get("skip_low_pano_search", True))
MIN_PANOS_PER_SEARCH = int(_cfg.get("min_panos_per_search", 2))

# Sequence-aware crawl tuning
SEQUENCE_CRAWL_TARGET_NEW = int(_cfg.get("sequence_crawl_target_new", RANDOM_CRAWL_TARGET_NEW))
SEQUENCE_CRAWL_MAX_SEARCHES = int(_cfg.get("sequence_crawl_max_searches", RANDOM_CRAWL_MAX_SEARCHES))
SEQUENCE_WALK_ENABLED = bool(_cfg.get("sequence_walk_enabled", True))
SEQUENCE_WALK_BIDIRECTIONAL = bool(_cfg.get("sequence_walk_bidirectional", True))
SEQUENCE_MAX_LENGTH = int(_cfg.get("sequence_max_length", 12))
SEQUENCE_STEP_METERS = float(_cfg.get("sequence_step_meters", 8.0))

# Legacy / random_incremental rows lack a true sequence; persist a sentinel
# so downstream consumers can distinguish them from real sequences.
LEGACY_SEARCH_POINT_ID = "unknown"
EARTH_RADIUS_METERS = 6_371_000.0

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
        log.info("%s exists, will upsert records", infoPath)


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
    """Load existing metadata, backfilling sequence-aware fields for legacy rows."""
    records: dict[str, dict] = {}
    if infoPath.exists():
        with open(infoPath, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if not row.get("pano_id"):
                    continue
                # Older crawls (pre-sequence schema) won't have these columns;
                # backfill so downstream code never has to None-guard them.
                if not row.get("timestamp"):
                    row["timestamp"] = ""
                if not row.get("search_point_id"):
                    row["search_point_id"] = LEGACY_SEARCH_POINT_ID
                records[row["pano_id"]] = row
    return records


def format_search_point(search_point: Tuple[float, float] | None) -> str:
    if search_point is None:
        return ""
    lat, lon = search_point
    return f"[{lat:.6f}, {lon:.6f}]"


def build_info_row(
    pano,
    search_point: Tuple[float, float] | None = None,
    *,
    search_point_id: str = LEGACY_SEARCH_POINT_ID,
    timestamp: str = "",
) -> dict:
    return {
        "pano_id": pano.pano_id,
        "lat": pano.lat,
        "lon": pano.lon,
        "heading": pano.heading,
        "pitch": pano.pitch,
        "roll": pano.roll,
        "date": pano.date,
        "search_point": format_search_point(search_point),
        "timestamp": timestamp,
        "search_point_id": search_point_id,
    }


def random_location(regions: list[tuple[float, float, float, float]] | None = None) -> Tuple[float, float]:
    lat_min, lat_max, lon_min, lon_max = random.choice(regions or DEFAULT_RANDOM_REGIONS)
    return random.uniform(lat_min, lat_max), random.uniform(lon_min, lon_max)


def download_missing_panorama(
    pano,
    records: dict[str, dict],
    session,
    search_point: Tuple[float, float] | None = None,
    *,
    search_point_id: str = LEGACY_SEARCH_POINT_ID,
    timestamp: str = "",
) -> bool:
    row = build_info_row(pano, search_point, search_point_id=search_point_id, timestamp=timestamp)
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
        if is_quota_error(e):
            log.warning("Google API daily soft limit reached, stopping crawl: %s", e)
            raise GoogleAPIQuotaExceededError(str(e)) from e
        log.warning("Search failed near (%.6f, %.6f): %s", lat, lon, e)
        return 0
    log.info("Found %d panorama(s)", len(panos))

    if not panos:
        log.warning("No panoramas found for the given location.")
        return 0

    if SKIP_LOW_PANO_SEARCH and len(panos) < MIN_PANOS_PER_SEARCH:
        log.info(
            "Skipping search point (%.6f, %.6f): only %d panorama(s), need >= %d",
            lat, lon, len(panos), MIN_PANOS_PER_SEARCH,
        )
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
                    write_info_records(records)
                    raise GoogleAPIQuotaExceededError(str(e)) from e
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

        if SKIP_LOW_PANO_SEARCH and len(panos) < MIN_PANOS_PER_SEARCH:
            log.info(
                "Skipping random search point (%.6f, %.6f): only %d panorama(s), need >= %d",
                lat, lon, len(panos), MIN_PANOS_PER_SEARCH,
            )
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


# ── Sequence-aware crawl ─────────────────────────────────────────────────────

def pick_sequence_cluster(panos: Iterable[Panorama]) -> list[Panorama]:
    """Group ``panos`` by ``date`` and return the group most likely to be a single
    capture session.

    Selection rule:
        1. Prefer the group with the most members (cross-year duplicates show up
           as separate one-element groups, so the largest group is the session).
        2. Tie-break by most recent ``date`` so we pick newer imagery when sizes
           match. Empty-date panos (``current``) are placed in their own bucket
           so we never mix them with dated history.

    Safety:
        ``pano.date == None`` means the *current/latest* sweep, not "unknown",
        so a *single* ``None`` pano in the response is normal — that's just
        the live version of one capture session. But **multiple** ``None``
        panos at one search point is a pollution signal: a single session
        yields exactly one current pano plus zero-or-more dated history
        versions. Multiple ``None`` panos therefore mean either
            (a) the API returned several distinct streets/sessions whose
                ``raw_dates`` entries got stripped, or
            (b) the response covers several adjacent streets each with their
                own current sweep, all collapsed under one search point.
        In either case the ``""`` group spans different sessions and must not
        be treated as one sequence. Whenever the ``""`` group has more than
        one member, fall back to an anchor-only cluster (a single row);
        ``_walk_sequence`` is unaffected and may still extend the sequence by
        finding current-session neighbors along the road.
    """
    groups: dict[str, list[Panorama]] = {}
    pano_list = list(panos)
    for pano in pano_list:
        key = pano.date or ""
        groups.setdefault(key, []).append(pano)
    if not groups:
        return []
    best_key = max(groups.keys(), key=lambda k: (len(groups[k]), k))
    # Anchor-only fallback whenever the winner is the undated bucket and it
    # contains more than one pano — see the Safety note above. This also
    # covers the all-undated case as a special instance.
    if best_key == "" and len(groups[""]) > 1:
        log.warning(
            "Sequence cluster: %d undated pano(s) at this search point "
            "(expected at most 1 current pano per session); "
            "falling back to anchor-only to avoid cross-session pollution",
            len(groups[""]),
        )
        return [groups[""][0]]
    return groups[best_key]


def step_lat_lon(lat: float, lon: float, heading_deg: float, distance_m: float) -> Tuple[float, float]:
    """Advance ``distance_m`` meters along ``heading_deg`` from ``(lat, lon)``.

    Uses the great-circle forward formula on a spherical Earth — accurate to a
    few centimeters at the 5-50m scales we walk along in sequence mode.
    """
    bearing = math.radians(heading_deg)
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    angular = distance_m / EARTH_RADIUS_METERS
    new_lat = math.asin(
        math.sin(lat_rad) * math.cos(angular)
        + math.cos(lat_rad) * math.sin(angular) * math.cos(bearing)
    )
    new_lon = lon_rad + math.atan2(
        math.sin(bearing) * math.sin(angular) * math.cos(lat_rad),
        math.cos(angular) - math.sin(lat_rad) * math.sin(new_lat),
    )
    return math.degrees(new_lat), math.degrees(new_lon)


def _select_sequence_neighbor(
    neighbors: Iterable[Panorama],
    *,
    anchor_date: str,
    seen_ids: set[str],
) -> Panorama | None:
    """Return the first neighbor pano sharing ``anchor_date`` that we haven't
    already absorbed into the current sequence.

    Note on ``None``-date semantics: in the GeoPhotoService response a missing
    date means **current / latest** capture, not "unknown". A current-session
    walk therefore legitimately matches ``None`` with ``None`` (both anchor
    and neighbor being the newest sweep along the same street).
    """
    anchor_key = anchor_date or ""
    for pano in neighbors:
        if pano.pano_id in seen_ids:
            continue
        if (pano.date or "") != anchor_key:
            continue
        return pano
    return None


def _walk_sequence(
    *,
    anchor: Panorama,
    initial_heading: float,
    search_point_id: str,
    records: dict[str, dict],
    session,
    seed_search_point: Tuple[float, float],
    max_extra: int,
    step_meters: float,
    target_new: int,
    current_added: int,
    seen_ids: set[str],
) -> int:
    """Extend a sequence by stepping along the road in a single direction.

    The first step is taken in ``initial_heading``; afterwards we follow each
    candidate's own heading so curved roads stay tracked. ``seen_ids`` is shared
    with the caller so a forward + backward walk pair never re-downloads the
    anchor or revisits its own discoveries.

    Returns the number of newly downloaded panoramas. Raises
    :class:`GoogleAPIQuotaExceededError` if the daily quota is hit so the
    caller can persist progress and stop.
    """
    if max_extra <= 0:
        return 0

    # ``anchor.date`` may be ``None`` — that means current/latest capture, not
    # missing metadata, and current-session walking along undated panos is a
    # valid same-session sequence (Street View routinely returns ``None`` for
    # the newest sweep).
    anchor_date = anchor.date or ""
    cur_lat, cur_lon = anchor.lat, anchor.lon
    heading = float(initial_heading)
    added_local = 0

    for _ in range(max_extra):
        next_lat, next_lon = step_lat_lon(cur_lat, cur_lon, heading, step_meters)
        try:
            neighbors = search_panoramas(lat=next_lat, lon=next_lon)
        except PanoramaSearchError as e:
            if is_quota_error(e):
                raise GoogleAPIQuotaExceededError(str(e)) from e
            log.warning("Sequence walk search failed near (%.6f, %.6f): %s", next_lat, next_lon, e)
            return added_local

        candidate = _select_sequence_neighbor(
            neighbors, anchor_date=anchor_date, seen_ids=seen_ids
        )
        if candidate is None:
            log.info("Sequence walk: no same-date neighbor at (%.6f, %.6f); stopping", next_lat, next_lon)
            return added_local

        seen_ids.add(candidate.pano_id)
        if candidate.pano_id in records and (panoPath / f"{candidate.pano_id}.png").exists():
            cur_lat, cur_lon = candidate.lat, candidate.lon
            heading = float(candidate.heading) if candidate.heading is not None else heading
            continue

        try:
            if download_missing_panorama(
                candidate,
                records,
                session,
                seed_search_point,
                search_point_id=search_point_id,
                timestamp=candidate.date or "",
            ):
                added_local += 1
                write_info_records(records)
                log.info(
                    "Sequence walk +1 (%d total): pano_id=%s search_point_id=%s heading=%.1f",
                    current_added + added_local, candidate.pano_id, search_point_id, heading,
                )
                if current_added + added_local >= target_new:
                    return added_local
        except Exception as e:
            if is_quota_error(e):
                raise GoogleAPIQuotaExceededError(str(e)) from e
            log.warning("  Sequence walk download failed [%s]: %s", candidate.pano_id, e)

        cur_lat, cur_lon = candidate.lat, candidate.lon
        # Roads bend — keep walking along the *candidate's* heading, not the
        # anchor's original direction. Fall back to previous heading if the
        # candidate has no heading set.
        heading = float(candidate.heading) if candidate.heading is not None else heading
        time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))

    return added_local


def fetch_random_sequence_panoramas(
    target_new: int = SEQUENCE_CRAWL_TARGET_NEW,
    max_searches: int = SEQUENCE_CRAWL_MAX_SEARCHES,
    *,
    walk_enabled: bool | None = None,
    walk_bidirectional: bool | None = None,
    max_sequence_length: int = SEQUENCE_MAX_LENGTH,
    step_meters: float = SEQUENCE_STEP_METERS,
) -> int:
    """Crawl panoramas grouped into capture sessions.

    Differences vs. ``fetch_random_incremental_panoramas``:
        * Cross-year duplicates at the same lat/lon are skipped — only the
          largest same-date group at each search point is downloaded.
        * Every record in a session shares one ``search_point_id`` (the anchor
          pano_id) and a ``timestamp`` (= its ``date`` field).
        * When ``walk_enabled`` is true the sequence is extended by stepping
          along the road. Each step adapts to the candidate pano's own heading
          so curved paths stay tracked.
        * When ``walk_bidirectional`` is true the walk is performed twice —
          once along ``anchor.heading`` (forward) and once along
          ``anchor.heading + 180°`` (backward) — sharing one ``seen_ids`` set
          to avoid revisits.
    """
    walk_enabled = SEQUENCE_WALK_ENABLED if walk_enabled is None else walk_enabled
    walk_bidirectional = SEQUENCE_WALK_BIDIRECTIONAL if walk_bidirectional is None else walk_bidirectional
    session = get_session()
    records = load_info_records()
    added = 0
    searches = 0

    while added < target_new and searches < max_searches:
        lat, lon = random_location()
        searches += 1
        log.info("Sequence search %d/%d near (%.6f, %.6f)", searches, max_searches, lat, lon)
        try:
            panos = search_panoramas(lat=lat, lon=lon)
        except PanoramaSearchError as e:
            if is_quota_error(e):
                log.warning("Google API daily soft limit reached, stopping crawl: %s", e)
                break
            log.warning("Search failed near (%.6f, %.6f): %s", lat, lon, e)
            continue

        if SKIP_LOW_PANO_SEARCH and len(panos) < MIN_PANOS_PER_SEARCH:
            log.info(
                "Skipping sequence search point (%.6f, %.6f): only %d panorama(s), need >= %d",
                lat, lon, len(panos), MIN_PANOS_PER_SEARCH,
            )
            continue

        cluster = pick_sequence_cluster(panos)
        if not cluster:
            continue

        anchor = cluster[0]
        search_point_id = anchor.pano_id
        log.info(
            "Sequence anchor pano_id=%s date=%s cluster_size=%d (skipped %d cross-date pano(s))",
            anchor.pano_id, anchor.date or "current", len(cluster), len(panos) - len(cluster),
        )

        try:
            for pano in cluster:
                if added >= target_new:
                    break
                if pano.pano_id in records and (panoPath / f"{pano.pano_id}.png").exists():
                    continue
                try:
                    if download_missing_panorama(
                        pano,
                        records,
                        session,
                        (lat, lon),
                        search_point_id=search_point_id,
                        timestamp=pano.date or "",
                    ):
                        added += 1
                        write_info_records(records)
                        log.info(
                            "Sequence crawl +1 (%d/%d): pano_id=%s search_point_id=%s",
                            added, target_new, pano.pano_id, search_point_id,
                        )
                except Exception as e:
                    if is_quota_error(e):
                        write_info_records(records)
                        log.warning("Google API daily soft limit reached, stopping crawl: %s", e)
                        return added
                    log.warning("  [%s] skipped: %s", pano.pano_id, e)
                if added < target_new:
                    time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))

            if walk_enabled and added < target_new:
                # seen_ids is shared across forward/backward walks so neither
                # direction re-downloads anchor or each other's discoveries.
                seen_ids: set[str] = {p.pano_id for p in cluster}
                base_heading = float(anchor.heading or 0.0)
                walk_directions = [base_heading]
                if walk_bidirectional:
                    walk_directions.append((base_heading + 180.0) % 360.0)

                remaining_budget = max(0, max_sequence_length - len(cluster))
                for direction in walk_directions:
                    if added >= target_new or remaining_budget <= 0:
                        break
                    walked = _walk_sequence(
                        anchor=anchor,
                        initial_heading=direction,
                        search_point_id=search_point_id,
                        records=records,
                        session=session,
                        seed_search_point=(lat, lon),
                        max_extra=remaining_budget,
                        step_meters=step_meters,
                        target_new=target_new,
                        current_added=added,
                        seen_ids=seen_ids,
                    )
                    added += walked
                    remaining_budget -= walked
        except GoogleAPIQuotaExceededError as e:
            write_info_records(records)
            log.warning("Google API daily soft limit reached, stopping crawl: %s", e)
            return added

    write_info_records(records)
    log.info("Sequence crawl added %d new panoramas after %d search(es)", added, searches)
    return added


# ── 入口 ─────────────────────────────────────────────────────────────────────

def main() -> int:
    init_info()
    if CRAWL_MODE == "random_incremental":
        fetch_random_incremental_panoramas()
    elif CRAWL_MODE == "random_search_sequence":
        fetch_random_sequence_panoramas()
    else:
        for location in locList:
            try:
                fetch_panoramas(location, isCurrent=False)
            except GoogleAPIQuotaExceededError as e:
                log.warning("Stopping crawl due to API quota: %s", e)
                break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
