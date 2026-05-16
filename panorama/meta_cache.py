"""JSON-backed cache for panorama metadata lookups.

Google's Street View Metadata endpoint counts toward our daily soft limit, so
sequence-aware crawling caches every successful response on disk and replays it
on subsequent runs. The cache is a flat ``{pano_id: entry}`` map persisted as
``runtime/pano_meta_cache.json`` by default.

Each entry stores the API ``date``/``location``/``pano_id`` plus a ``cached_at``
unix timestamp so callers can opt into a TTL.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

from .api import MetaData, get_panorama_meta
from .config import cfg as _cfg, resolve_project_path
from .quota import GoogleAPIQuotaExceededError, reserve_request

CACHE_PATH = resolve_project_path(_cfg.get("metadata_cache_path", "runtime/pano_meta_cache.json"))
CACHE_TTL_SECONDS = int(_cfg.get("metadata_cache_ttl_seconds", 30 * 24 * 3600))


def _load(cache_path: Path) -> dict[str, Any]:
    if not cache_path.exists():
        return {}
    try:
        with open(cache_path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _save(cache_path: Path, cache: dict[str, Any]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2, sort_keys=True)
    tmp.replace(cache_path)


def _entry_to_meta(entry: dict[str, Any]) -> MetaData:
    return MetaData(
        pano_id=entry["pano_id"],
        date=entry.get("date"),
        location={"lat": entry["lat"], "lng": entry["lng"]},
    )


def _meta_to_entry(meta: MetaData, now: int) -> dict[str, Any]:
    return {
        "pano_id": meta.pano_id,
        "date": meta.date,
        "lat": meta.location.lat,
        "lng": meta.location.lng,
        "cached_at": now,
    }


def cached_get_panorama_meta(
    pano_id: str,
    api_key: str,
    *,
    cache_path: Optional[Path] = None,
    ttl_seconds: Optional[int] = None,
    now: Optional[int] = None,
) -> MetaData:
    """Return the metadata for ``pano_id`` from cache or the live API.

    Reserves one ``metadata_requests`` quota slot only on cache miss. Raises
    :class:`GoogleAPIQuotaExceededError` if the soft limit is reached before
    the live call is attempted.
    """
    cache_path = cache_path or CACHE_PATH
    ttl_seconds = CACHE_TTL_SECONDS if ttl_seconds is None else ttl_seconds
    now = int(time.time()) if now is None else int(now)

    cache = _load(cache_path)
    entry = cache.get(pano_id)
    if entry and now - int(entry.get("cached_at", 0)) < ttl_seconds:
        try:
            return _entry_to_meta(entry)
        except (KeyError, TypeError, ValueError):
            # Fall through and refetch on malformed entries.
            cache.pop(pano_id, None)

    try:
        reserve_request("metadata_requests")
    except GoogleAPIQuotaExceededError:
        raise

    meta = get_panorama_meta(pano_id, api_key)
    cache[pano_id] = _meta_to_entry(meta, now)
    _save(cache_path, cache)
    return meta


def clear_cache(cache_path: Optional[Path] = None) -> None:
    cache_path = cache_path or CACHE_PATH
    if cache_path.exists():
        cache_path.unlink()
