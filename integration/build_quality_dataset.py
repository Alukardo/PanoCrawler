import argparse
import csv
import json
import logging
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from panorama.config import cfg, resolve_project_path
from panorama.download import PanoDownloadError, get_panorama_stages, get_session
from panorama.quality import bottom_black_edge_ratio, build_quality_metrics
from panorama.search import Panorama, PanoramaSearchError, search_panoramas

log = logging.getLogger(__name__)

URBAN_BOUNDS = [
    ("taipei", 25.0200, 25.0800, 121.4800, 121.5700),
    ("tokyo", 35.6400, 35.7200, 139.6900, 139.7900),
    ("new_york", 40.7000, 40.7900, -74.0200, -73.9300),
    ("london", 51.4900, 51.5300, -0.1600, -0.0700),
    ("paris", 48.8400, 48.8800, 2.3000, 2.3800),
    ("berlin", 52.4900, 52.5400, 13.3500, 13.4500),
    ("singapore", 1.2800, 1.3500, 103.8000, 103.8800),
    ("sydney", -33.8900, -33.8400, 151.1800, 151.2400),
    ("san_francisco", 37.7600, 37.8100, -122.4500, -122.3900),
    ("toronto", 43.6300, 43.6800, -79.4200, -79.3500),
    ("seoul", 37.5400, 37.5900, 126.9400, 127.0300),
    ("hong_kong", 22.2700, 22.3300, 114.1400, 114.2200),
]


@dataclass(frozen=True)
class Candidate:
    city: str
    sample_lat: float
    sample_lon: float
    pano: Panorama

    @property
    def year(self) -> str:
        return self.pano.date[:4] if self.pano.date else "current"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--min-years", type=int, default=4)
    parser.add_argument("--max-search-attempts", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--zoom", type=int, default=None)
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--output-dir", default="quality_dataset")
    parser.add_argument("--include-current", action="store_true")
    parser.add_argument("--max-bottom-black-ratio", type=float, default=0.01)
    parser.add_argument("--min-sharpness-ratio", type=float, default=0.95)
    parser.add_argument("--max-heading-diff", type=float, default=0.0)
    parser.add_argument("--fail-on-quality", action="store_true")
    return parser.parse_args()


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


def random_location(rng: random.Random) -> tuple[str, float, float]:
    city, min_lat, max_lat, min_lon, max_lon = rng.choice(URBAN_BOUNDS)
    return city, rng.uniform(min_lat, max_lat), rng.uniform(min_lon, max_lon)


def select_diverse_candidates(candidates: list[Candidate], sample_count: int) -> list[Candidate]:
    selected: list[Candidate] = []
    selected_ids: set[str] = set()
    years = sorted({candidate.year for candidate in candidates})
    for year in years:
        for candidate in candidates:
            if candidate.year == year and candidate.pano.pano_id not in selected_ids:
                selected.append(candidate)
                selected_ids.add(candidate.pano.pano_id)
                break
        if len(selected) >= sample_count:
            return selected
    for candidate in candidates:
        if candidate.pano.pano_id in selected_ids:
            continue
        selected.append(candidate)
        selected_ids.add(candidate.pano.pano_id)
        if len(selected) >= sample_count:
            break
    return selected


def collect_candidates(args: argparse.Namespace) -> list[Candidate]:
    rng = random.Random(args.seed)
    candidates: list[Candidate] = []
    seen: set[str] = set()
    for attempt in range(args.max_search_attempts):
        city, lat, lon = random_location(rng)
        try:
            panos = search_panoramas(lat=lat, lon=lon)
        except PanoramaSearchError as e:
            log.warning("search failed city=%s lat=%.6f lon=%.6f error=%s", city, lat, lon, e)
            continue
        log.info("search %d/%d city=%s lat=%.6f lon=%.6f panos=%d", attempt + 1, args.max_search_attempts, city, lat, lon, len(panos))
        for pano in panos:
            if pano.pano_id in seen:
                continue
            if pano.date is None and not args.include_current:
                continue
            seen.add(pano.pano_id)
            candidates.append(Candidate(city=city, sample_lat=lat, sample_lon=lon, pano=pano))
        selected = select_diverse_candidates(candidates, args.samples)
        years = {candidate.year for candidate in selected if candidate.year != "current"}
        if len(selected) >= args.samples and len(years) >= args.min_years:
            return selected
    return select_diverse_candidates(candidates, args.samples)


def quality_pass(row: dict[str, Any], args: argparse.Namespace) -> bool:
    return (
        row["final_bottom_black_ratio"] <= args.max_bottom_black_ratio
        and row["sharpness_ratio"] >= args.min_sharpness_ratio
        and row["heading_mean_abs_diff"] <= args.max_heading_diff
    )


def write_dataset(args: argparse.Namespace, candidates: list[Candidate]) -> list[dict[str, Any]]:
    output_dir = resolve_project_path(args.output_dir)
    raw_dir = output_dir / "raw"
    base_dir = output_dir / "base"
    final_dir = output_dir / "final"
    for directory in [raw_dir, base_dir, final_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.csv"
    metrics_path = output_dir / "metrics.jsonl"
    session = get_session()
    rows: list[dict[str, Any]] = []
    fieldnames = [
        "status",
        "quality_pass",
        "city",
        "sample_lat",
        "sample_lon",
        "pano_id",
        "pano_lat",
        "pano_lon",
        "date",
        "year",
        "heading",
        "pitch",
        "roll",
        "zoom",
        "downloaded_tiles",
        "raw_path",
        "base_path",
        "final_path",
        "raw_bottom_black_ratio",
        "final_bottom_black_ratio",
        "base_sharpness",
        "final_sharpness",
        "sharpness_ratio",
        "heading_shift_px",
        "heading_mean_abs_diff",
        "error",
    ]
    with manifest_path.open("w", encoding="utf-8", newline="") as csv_file, metrics_path.open("w", encoding="utf-8") as jsonl_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for index, candidate in enumerate(candidates, start=1):
            pano = candidate.pano
            stem = f"{candidate.year}_{safe_name(pano.pano_id)}"
            raw_path = raw_dir / f"{stem}.png"
            base_path = base_dir / f"{stem}.png"
            final_path = final_dir / f"{stem}.png"
            row: dict[str, Any] = {
                "status": "failed",
                "quality_pass": False,
                "city": candidate.city,
                "sample_lat": candidate.sample_lat,
                "sample_lon": candidate.sample_lon,
                "pano_id": pano.pano_id,
                "pano_lat": pano.lat,
                "pano_lon": pano.lon,
                "date": pano.date,
                "year": candidate.year,
                "heading": pano.heading,
                "pitch": pano.pitch,
                "roll": pano.roll,
                "zoom": args.zoom,
                "downloaded_tiles": 0,
                "raw_path": str(raw_path),
                "base_path": str(base_path),
                "final_path": str(final_path),
                "raw_bottom_black_ratio": None,
                "final_bottom_black_ratio": None,
                "base_sharpness": None,
                "final_sharpness": None,
                "sharpness_ratio": None,
                "heading_shift_px": None,
                "heading_mean_abs_diff": None,
                "error": "",
            }
            try:
                log.info("download %d/%d pano_id=%s date=%s city=%s", index, len(candidates), pano.pano_id, pano.date, candidate.city)
                stages = get_panorama_stages(pano=pano, zoom=args.zoom, session=session)
                stages.raw.save(raw_path, "PNG")
                stages.base.save(base_path, "PNG")
                stages.final.save(final_path, "PNG")
                metrics = build_quality_metrics(stages.base, stages.final, pano.heading, threshold=cfg.get("black_threshold", 15))
                metric_data = metrics.to_dict()
                row["status"] = "ok"
                row["zoom"] = stages.zoom
                row["downloaded_tiles"] = stages.downloaded_tiles
                row["raw_bottom_black_ratio"] = bottom_black_edge_ratio(stages.raw, threshold=cfg.get("black_threshold", 15))
                row["final_bottom_black_ratio"] = metric_data["bottom_black_ratio"]
                row["base_sharpness"] = metric_data["base_sharpness"]
                row["final_sharpness"] = metric_data["final_sharpness"]
                row["sharpness_ratio"] = metric_data["sharpness_ratio"]
                row["heading_shift_px"] = metric_data["heading_shift_px"]
                row["heading_mean_abs_diff"] = metric_data["heading_mean_abs_diff"]
                row["quality_pass"] = quality_pass(row, args)
            except (PanoDownloadError, ValueError, OSError) as e:
                row["error"] = str(e)
                log.warning("failed pano_id=%s error=%s", pano.pano_id, e)
            writer.writerow(row)
            jsonl_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            jsonl_file.flush()
            rows.append(row)
            if index < len(candidates):
                time.sleep(args.delay)
    log.info("manifest: %s", manifest_path)
    log.info("metrics: %s", metrics_path)
    return rows


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    if not os.getenv("GOOGLE_API_KEY"):
        raise SystemExit("GOOGLE_API_KEY is required. Put it in .apikey or the environment.")
    candidates = collect_candidates(args)
    years = sorted({candidate.year for candidate in candidates})
    log.info("selected candidates=%d years=%s", len(candidates), ",".join(years))
    if not candidates:
        raise SystemExit("No panorama candidates found.")
    rows = write_dataset(args, candidates)
    failed = [row for row in rows if row["status"] != "ok"]
    quality_failed = [row for row in rows if row["status"] == "ok" and not row["quality_pass"]]
    log.info("completed ok=%d failed=%d quality_failed=%d", len(rows) - len(failed), len(failed), len(quality_failed))
    if args.fail_on_quality and quality_failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
