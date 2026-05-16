"""Diagnostic CLI for sequence-aware panorama metadata.

Groups ``info.csv`` rows by ``search_point_id``, sorts each group by capture order
(falling back to lat/lon-based nearest-neighbor when order is missing), reports
contiguity stats, and flags suspicious gaps so you can decide whether to keep
or rebuild a sequence.

Usage:
    python -m integration.sequence_audit                       # uses config metadata_path
    python -m integration.sequence_audit --input path/info.csv
    python -m integration.sequence_audit --json                # emit machine-readable JSON
    python -m integration.sequence_audit --gap-threshold-meters 30
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from panorama.config import cfg as _cfg, resolve_images_path

LEGACY_SEARCH_POINT_ID = "unknown"
EARTH_RADIUS_METERS = 6_371_000.0
DEFAULT_GAP_THRESHOLD_METERS = 30.0


@dataclass
class SequenceMember:
    pano_id: str
    lat: float
    lon: float
    heading: float | None
    date: str
    order: int


@dataclass
class SequenceReport:
    search_point_id: str
    members: list[SequenceMember] = field(default_factory=list)
    date: str = ""

    @property
    def length(self) -> int:
        return len(self.members)


def haversine_meters(a_lat: float, a_lon: float, b_lat: float, b_lon: float) -> float:
    """Great-circle distance in meters between two lat/lon points.

    Uses the haversine formula on a spherical Earth model. Accurate to a few
    meters at the scales we care about (street-level segments).
    """
    a_lat_r = math.radians(a_lat)
    b_lat_r = math.radians(b_lat)
    d_lat = math.radians(b_lat - a_lat)
    d_lon = math.radians(b_lon - a_lon)
    h = math.sin(d_lat / 2) ** 2 + math.cos(a_lat_r) * math.cos(b_lat_r) * math.sin(d_lon / 2) ** 2
    return 2 * EARTH_RADIUS_METERS * math.asin(math.sqrt(h))


def load_sequences(csv_path: Path) -> dict[str, SequenceReport]:
    """Read ``info.csv`` and group rows by ``search_point_id``.

    Each member's ``order`` is its row position so we can reconstruct walk
    order (sequence mode emits members in the order they were downloaded).
    """
    sequences: dict[str, SequenceReport] = {}
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            pano_id = (row.get("pano_id") or "").strip()
            if not pano_id:
                continue
            search_point_id = (row.get("search_point_id") or LEGACY_SEARCH_POINT_ID).strip() or LEGACY_SEARCH_POINT_ID
            try:
                lat = float(row["lat"])
                lon = float(row["lon"])
            except (KeyError, TypeError, ValueError):
                continue
            heading_raw = (row.get("heading") or "").strip()
            try:
                heading = float(heading_raw) if heading_raw else None
            except ValueError:
                heading = None

            member = SequenceMember(
                pano_id=pano_id,
                lat=lat,
                lon=lon,
                heading=heading,
                date=(row.get("date") or "").strip(),
                order=idx,
            )
            report = sequences.setdefault(search_point_id, SequenceReport(search_point_id=search_point_id))
            report.members.append(member)
            if not report.date and member.date:
                report.date = member.date
    return sequences


def sort_members(members: list[SequenceMember]) -> list[SequenceMember]:
    """Return ``members`` in walk order (by ``order``) — preserves the original
    discovery sequence emitted by ``fetch_random_sequence_panoramas``."""
    return sorted(members, key=lambda m: m.order)


def step_distances(members: list[SequenceMember]) -> list[float]:
    """Pairwise haversine distances between consecutive members (meters)."""
    distances: list[float] = []
    for prev, curr in zip(members, members[1:]):
        distances.append(haversine_meters(prev.lat, prev.lon, curr.lat, curr.lon))
    return distances


def summarize_sequence(report: SequenceReport, gap_threshold_m: float) -> dict:
    """Build a dict of contiguity statistics for one sequence."""
    ordered = sort_members(report.members)
    distances = step_distances(ordered)
    lats = [m.lat for m in ordered]
    lons = [m.lon for m in ordered]
    bbox = {
        "lat_min": min(lats) if lats else None,
        "lat_max": max(lats) if lats else None,
        "lon_min": min(lons) if lons else None,
        "lon_max": max(lons) if lons else None,
    }
    gap_count = sum(1 for d in distances if d > gap_threshold_m)
    return {
        "search_point_id": report.search_point_id,
        "length": report.length,
        "date": report.date,
        "step_count": len(distances),
        "mean_step_m": sum(distances) / len(distances) if distances else 0.0,
        "median_step_m": sorted(distances)[len(distances) // 2] if distances else 0.0,
        "max_step_m": max(distances) if distances else 0.0,
        "min_step_m": min(distances) if distances else 0.0,
        "gap_count": gap_count,
        "is_contiguous": gap_count == 0 and report.length > 1,
        "bbox": bbox,
        "pano_ids": [m.pano_id for m in ordered],
    }


def build_report(
    sequences: dict[str, SequenceReport],
    gap_threshold_m: float,
) -> dict:
    """Aggregate per-sequence summaries plus global counts."""
    real = {sid: rep for sid, rep in sequences.items() if sid != LEGACY_SEARCH_POINT_ID}
    unknown = sequences.get(LEGACY_SEARCH_POINT_ID)

    summaries = [summarize_sequence(rep, gap_threshold_m) for rep in real.values()]
    summaries.sort(key=lambda s: (-s["length"], s["search_point_id"]))

    real_panos = sum(s["length"] for s in summaries)
    contiguous = sum(1 for s in summaries if s["is_contiguous"])
    gapped = sum(1 for s in summaries if s["gap_count"] > 0)
    singletons = sum(1 for s in summaries if s["length"] == 1)

    return {
        "gap_threshold_m": gap_threshold_m,
        "totals": {
            "sequences": len(summaries),
            "panoramas_in_sequences": real_panos,
            "panoramas_unknown": unknown.length if unknown else 0,
            "contiguous_sequences": contiguous,
            "gapped_sequences": gapped,
            "singleton_sequences": singletons,
        },
        "sequences": summaries,
    }


def format_text_report(report: dict) -> str:
    """Pretty-print the report for terminals."""
    totals = report["totals"]
    lines = [
        "Sequence Audit",
        "=" * 60,
        f"Gap threshold       : {report['gap_threshold_m']:.1f} m",
        f"Total sequences     : {totals['sequences']}",
        f"  contiguous        : {totals['contiguous_sequences']}",
        f"  with gaps         : {totals['gapped_sequences']}",
        f"  singletons        : {totals['singleton_sequences']}",
        f"Panoramas in seq.   : {totals['panoramas_in_sequences']}",
        f"Panoramas legacy    : {totals['panoramas_unknown']} (search_point_id='{LEGACY_SEARCH_POINT_ID}')",
        "",
        f"{'search_point_id':32s} {'len':>4s} {'date':>8s} {'mean_m':>8s} {'max_m':>8s} {'gaps':>5s}  status",
        "-" * 78,
    ]
    for s in report["sequences"]:
        status = "OK" if s["is_contiguous"] else ("SINGLE" if s["length"] == 1 else "GAPPED")
        lines.append(
            f"{s['search_point_id'][:32]:32s} "
            f"{s['length']:>4d} "
            f"{(s['date'] or '-'):>8s} "
            f"{s['mean_step_m']:>8.1f} "
            f"{s['max_step_m']:>8.1f} "
            f"{s['gap_count']:>5d}  "
            f"{status}"
        )
    return "\n".join(lines)


def run_audit(
    input_path: Path,
    gap_threshold_m: float = DEFAULT_GAP_THRESHOLD_METERS,
) -> dict:
    """Public API used by tests; returns the structured report dict."""
    sequences = load_sequences(input_path)
    return build_report(sequences, gap_threshold_m)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit search_point_id grouping in panorama info.csv")
    parser.add_argument(
        "--input",
        type=Path,
        default=resolve_images_path(_cfg.get("metadata_path", "info.csv")),
        help="Path to info.csv (default: config metadata_path)",
    )
    parser.add_argument(
        "--gap-threshold-meters",
        type=float,
        default=DEFAULT_GAP_THRESHOLD_METERS,
        help=f"Flag step distances above this as gaps (default: {DEFAULT_GAP_THRESHOLD_METERS})",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if not args.input.exists():
        print(f"error: metadata file not found: {args.input}", file=sys.stderr)
        return 1

    report = run_audit(args.input, args.gap_threshold_meters)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(format_text_report(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
