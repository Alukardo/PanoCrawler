import json
import os
import fcntl
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable

from .config import cfg as _cfg, resolve_project_path


COUNTED_REQUEST_KEYS = {"search_requests", "session_requests", "tile_requests"}
USAGE_PATH = resolve_project_path(_cfg.get("google_api_usage_path", "runtime/google_api_usage.json"))
DAILY_QUOTA = int(_cfg.get("google_api_daily_quota", 15000))
DAILY_SOFT_LIMIT = int(_cfg.get("google_api_daily_soft_limit", min(DAILY_QUOTA, 13500)))


class GoogleAPIQuotaExceededError(Exception):
    pass


class GoogleAPIUsageError(Exception):
    pass


def tracking_enabled() -> bool:
    return (
        _cfg.get("google_api_quota_tracking_enabled", True)
        and os.getenv("PANOCRAWLER_DISABLE_QUOTA") != "1"
        and os.getenv("PYTEST_CURRENT_TEST") is None
    )


def today_key() -> str:
    return date.today().isoformat()


def load_usage(usage_path: Path | None = None) -> dict[str, Any]:
    usage_path = usage_path or USAGE_PATH
    if not usage_path.exists():
        return {}
    try:
        with open(usage_path, encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise GoogleAPIUsageError(f"Google API usage file is corrupted: {usage_path}") from e
    except OSError as e:
        raise GoogleAPIUsageError(f"Could not read Google API usage file: {usage_path}") from e


def save_usage(usage: dict[str, Any], usage_path: Path | None = None) -> None:
    usage_path = usage_path or USAGE_PATH
    usage_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = usage_path.with_suffix(usage_path.suffix + ".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(usage, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    temp_path.replace(usage_path)


def update_usage(
    updater: Callable[[dict[str, Any]], dict[str, Any]],
    usage_path: Path | None = None,
) -> dict[str, Any]:
    usage_path = usage_path or USAGE_PATH
    usage_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = usage_path.with_suffix(usage_path.suffix + ".lock")
    with open(lock_path, "w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        usage = load_usage(usage_path)
        result = updater(usage)
        save_usage(usage, usage_path)
        fcntl.flock(lock_file, fcntl.LOCK_UN)
        return result


def estimated_total(day_usage: dict[str, Any]) -> int:
    return sum(int(day_usage.get(key, 0)) for key in COUNTED_REQUEST_KEYS)


def get_day_usage(usage: dict[str, Any], day: str) -> dict[str, Any]:
    if day not in usage or not isinstance(usage[day], dict):
        usage[day] = {}
    return usage[day]


def reserve_request(
    category: str,
    amount: int = 1,
    usage_path: Path | None = None,
    day: str | None = None,
    soft_limit: int | None = None,
) -> dict[str, Any]:
    if not tracking_enabled():
        return {}
    day = day or today_key()
    soft_limit = DAILY_SOFT_LIMIT if soft_limit is None else soft_limit

    def updater(usage: dict[str, Any]) -> dict[str, Any]:
        day_usage = get_day_usage(usage, day)
        current_total = estimated_total(day_usage)
        if category in COUNTED_REQUEST_KEYS and current_total + amount > soft_limit:
            raise GoogleAPIQuotaExceededError(
                f"Google API daily soft limit reached: {current_total}/{soft_limit} requests used for {day}"
            )
        day_usage[category] = int(day_usage.get(category, 0)) + amount
        day_usage["estimated_total"] = estimated_total(day_usage)
        day_usage["daily_quota"] = DAILY_QUOTA
        day_usage["daily_soft_limit"] = soft_limit
        day_usage["updated_at"] = datetime.now().isoformat(timespec="seconds")
        return day_usage

    return update_usage(updater, usage_path)


def record_failed_request(
    category: str = "failed_requests",
    amount: int = 1,
    usage_path: Path | None = None,
    day: str | None = None,
) -> dict[str, Any]:
    if not tracking_enabled():
        return {}
    day = day or today_key()

    def updater(usage: dict[str, Any]) -> dict[str, Any]:
        day_usage = get_day_usage(usage, day)
        day_usage[category] = int(day_usage.get(category, 0)) + amount
        day_usage["estimated_total"] = estimated_total(day_usage)
        day_usage["daily_quota"] = DAILY_QUOTA
        day_usage["daily_soft_limit"] = DAILY_SOFT_LIMIT
        day_usage["updated_at"] = datetime.now().isoformat(timespec="seconds")
        return day_usage

    return update_usage(updater, usage_path)


def get_today_usage(usage_path: Path | None = None) -> dict[str, Any]:
    usage = load_usage(usage_path)
    return dict(usage.get(today_key(), {}))
