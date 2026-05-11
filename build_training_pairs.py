"""
Build paired panorama training samples for GAN-style models.

Each valid geo-location pair produces two directional samples:
  A→B  and  B→A  (reverse direction, negated offsets).
"""

import csv
import itertools
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol, Sequence

from panorama.config import cfg as _cfg, resolve_images_path, resolve_project_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Load config ───────────────────────────────────────────────────────────────

# ── Paths ─────────────────────────────────────────────────────────────────────
SOURCE_DIR = resolve_images_path(_cfg.get("images_dir", "pano"))
METADATA_PATH = resolve_images_path(_cfg.get("metadata_path", "info.csv"))
OUT_DIR = resolve_project_path(_cfg["temp_dir"])
INPUT_DIR = OUT_DIR / "train_A"
OUTPUT_DIR = OUT_DIR / "train_B"
LABEL_DIR = OUT_DIR / "train_cond"

# 距离阈值：小于此值视为同位置（单位：度²）
DISTANCE_THRESHOLD = _cfg["distance_threshold"]  # default 1e-8 ≈ 11m at equator


@dataclass(frozen=True)
class PanoramaRecord:
    pano_id: str
    lat: float
    lon: float
    heading: float | None = None
    pitch: float | None = None
    roll: float | None = None
    date: str | None = None

    @classmethod
    def from_row(cls, row: dict[str, str | None]) -> "PanoramaRecord":
        pano_id = (row["pano_id"] or "").strip()
        if not pano_id:
            raise ValueError("missing pano_id")
        return cls(
            pano_id=pano_id,
            lat=float((row["lat"] or "").strip()),
            lon=float((row["lon"] or "").strip()),
            heading=parse_optional_float(row.get("heading")),
            pitch=parse_optional_float(row.get("pitch")),
            roll=parse_optional_float(row.get("roll")),
            date=(row.get("date") or "").strip() or None,
        )


def parse_optional_float(value: str | None) -> float | None:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    return float(value)


class DistanceMetric(Protocol):
    def is_close(self, a: PanoramaRecord, b: PanoramaRecord) -> bool:
        ...


class SquaredDegreeDistance:
    def __init__(self, threshold: float):
        self.threshold = threshold

    def is_close(self, a: PanoramaRecord, b: PanoramaRecord) -> bool:
        d_lat = a.lat - b.lat
        d_lon = a.lon - b.lon
        return d_lat * d_lat + d_lon * d_lon < self.threshold


class PairSelector(Protocol):
    def select(self, records: Sequence[PanoramaRecord]) -> Iterable[tuple[PanoramaRecord, PanoramaRecord]]:
        ...


class AllPairsWithinDistance:
    def __init__(self, distance_metric: DistanceMetric):
        self.distance_metric = distance_metric

    def select(self, records: Sequence[PanoramaRecord]) -> Iterable[tuple[PanoramaRecord, PanoramaRecord]]:
        for a, b in itertools.combinations(records, 2):
            if self.distance_metric.is_close(a, b):
                yield a, b


class SampleWriter(Protocol):
    def prepare(self) -> None:
        ...

    def write_pair(self, a: PanoramaRecord, b: PanoramaRecord, start_index: int) -> int:
        """Write samples for a pair and return the number of samples written."""
        ...


class CopyBidirectionalSampleWriter:
    def __init__(
        self,
        source_dir: Path,
        input_dir: Path,
        output_dir: Path,
        label_dir: Path,
        prefix_int: str,
        prefix_out: str,
        prefix_ins: str,
    ):
        self.source_dir = source_dir
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.label_dir = label_dir
        self.prefix_int = prefix_int
        self.prefix_out = prefix_out
        self.prefix_ins = prefix_ins

    def prepare(self) -> None:
        for d in [self.input_dir, self.output_dir, self.label_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def write_pair(self, a: PanoramaRecord, b: PanoramaRecord, start_index: int) -> int:
        d_lat = a.lat - b.lat
        d_lon = a.lon - b.lon
        source = self.source_dir / f"{a.pano_id}.png"
        target = self.source_dir / f"{b.pano_id}.png"

        # Skip if either source file is missing
        if not source.exists():
            log.warning("Source file not found, skipping pair: %s", source.name)
            return 0
        if not target.exists():
            log.warning("Target file not found, skipping pair: %s", target.name)
            return 0

        source_bytes = source.read_bytes()
        target_bytes = target.read_bytes()

        # Forward pair: a → b
        (self.input_dir / f"{self.prefix_int}{start_index:05d}.png").write_bytes(source_bytes)
        (self.output_dir / f"{self.prefix_out}{start_index:05d}.png").write_bytes(target_bytes)
        (self.label_dir / f"{self.prefix_ins}{start_index:05d}.txt").write_text(f"{d_lat:.16f}\n{d_lon:.16f}\n")
        next_index = start_index + 1

        # Reverse pair: b → a
        (self.input_dir / f"{self.prefix_int}{next_index:05d}.png").write_bytes(target_bytes)
        (self.output_dir / f"{self.prefix_out}{next_index:05d}.png").write_bytes(source_bytes)
        (self.label_dir / f"{self.prefix_ins}{next_index:05d}.txt").write_text(f"{-d_lat:.16f}\n{-d_lon:.16f}\n")
        return 2


def remove_files_in_directory(dic: Path) -> None:
    """Remove regular files in a directory without recursing into subdirectories."""
    if not dic.exists():
        return
    for file_name in os.listdir(dic):
        file_path = dic / file_name
        if file_path.is_file():
            file_path.unlink()


def clean(dic: Path) -> None:
    """Remove all files in a directory without removing the directory itself."""
    remove_files_in_directory(dic)


def clean_output_dirs(dirs: Sequence[Path] = (INPUT_DIR, OUTPUT_DIR, LABEL_DIR)) -> None:
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        clean(d)


def load_records(filename: Path) -> list[PanoramaRecord]:
    records: list[PanoramaRecord] = []
    with open(filename, encoding="utf-8") as csvfile:
        csv_reader = csv.DictReader(csvfile)
        for row_number, row in enumerate(csv_reader, start=2):
            try:
                records.append(PanoramaRecord.from_row(row))
            except (KeyError, TypeError, ValueError) as e:
                log.warning("Skipping invalid metadata row %d pano_id=%r: %s", row_number, row.get("pano_id"), e)
    return records


def build_training_pairs(
    filename: Path,
    pair_selector: PairSelector | None = None,
    sample_writer: SampleWriter | None = None,
) -> int:
    """
    Generate paired training samples from panorama info CSV.

    Only pairs with the same location (distance < DISTANCE_THRESHOLD) are kept.
    Each valid pair produces two samples (A→B and B→A).

    Returns:
        Number of training pairs generated.
    """
    records = load_records(filename)
    pair_selector = pair_selector or AllPairsWithinDistance(SquaredDegreeDistance(DISTANCE_THRESHOLD))
    sample_writer = sample_writer or CopyBidirectionalSampleWriter(
        source_dir=SOURCE_DIR,
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        label_dir=LABEL_DIR,
        prefix_int=_cfg["prefix_int"],
        prefix_out=_cfg["prefix_out"],
        prefix_ins=_cfg["prefix_ins"],
    )
    sample_writer.prepare()
    count = 0

    for a, b in pair_selector.select(records):
        count += sample_writer.write_pair(a, b, count)

    return count


def process_data(
    filename: Path,
    pair_selector: PairSelector | None = None,
    sample_writer: SampleWriter | None = None,
) -> int:
    """Backward-compatible wrapper for build_training_pairs."""
    return build_training_pairs(filename, pair_selector=pair_selector, sample_writer=sample_writer)


if __name__ == "__main__":
    clean_output_dirs()
    pairs = build_training_pairs(METADATA_PATH)
    log.info("Training pairs generated: %d", pairs)
    log.info("Output: %s", INPUT_DIR.parent)
