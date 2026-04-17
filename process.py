"""
Panorama Data Processing — generates paired training samples for GAN-style models.

Each valid geo-location pair produces two directional samples:
  A→B  and  B→A  (reverse direction, negated offsets).
"""

import csv
import itertools
import logging
import os
from pathlib import Path
from typing import List

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
SOURCE_DIR = Path(__file__).parent.parent / "images" / "pano"
OUT_DIR = Path(__file__).parent.parent / "temp"
INPUT_DIR = OUT_DIR / "train_A"
OUTPUT_DIR = OUT_DIR / "train_B"
LABEL_DIR = OUT_DIR / "train_cond"

# 距离阈值：小于此值视为同位置（单位：度²）
# 0.00000001 ≈ 1.1m at equator
# NOTE: 此阈值较严格。Google 全景 GPS 存在误差，同一地点两次拍摄
#       可能略超此值而导致配对失败。如配对数量过少，可适当放宽。
DISTANCE_THRESHOLD = 0.00000001


def clean(dic: Path) -> None:
    """Remove all files in a directory without removing the directory itself."""
    if not dic.exists():
        return
    for file_name in os.listdir(dic):
        file_path = dic / file_name
        if file_path.is_file():
            file_path.unlink()


def process_data(filename: Path) -> int:
    """
    Generate paired training samples from panorama info CSV.

    Only pairs with the same location (distance < DISTANCE_THRESHOLD) are kept.
    Each valid pair produces two samples (A→B and B→A).

    Returns:
        Number of training pairs generated.
    """
    data: List[dict] = []

    with open(filename, encoding="utf-8") as csvfile:
        csv_reader = csv.DictReader(csvfile)
        for row in csv_reader:
            data.append(row)

    pairs = list(itertools.combinations(data, 2))
    count = 0

    for a, b in pairs:
        d_lat = float(a["lat"]) - float(b["lat"])
        d_lon = float(a["lon"]) - float(b["lon"])

        if d_lat * d_lat + d_lon * d_lon >= DISTANCE_THRESHOLD:
            continue

        source = SOURCE_DIR / f"{a['pano_id']}.png"
        target = SOURCE_DIR / f"{b['pano_id']}.png"

        # Skip if either source file is missing
        if not source.exists():
            log.warning("Source file not found, skipping pair: %s", source.name)
            continue
        if not target.exists():
            log.warning("Target file not found, skipping pair: %s", target.name)
            continue

        # Forward pair: a → b
        (INPUT_DIR / f"int{count:05d}.png").write_bytes(source.read_bytes())
        (OUTPUT_DIR / f"out{count:05d}.png").write_bytes(target.read_bytes())
        (LABEL_DIR / f"ins{count:05d}.txt").write_text(f"{d_lat:.16f}\n{d_lon:.16f}\n")
        count += 1

        # Reverse pair: b → a
        (INPUT_DIR / f"int{count:05d}.png").write_bytes(target.read_bytes())
        (OUTPUT_DIR / f"out{count:05d}.png").write_bytes(source.read_bytes())
        (LABEL_DIR / f"ins{count:05d}.txt").write_text(f"{-d_lat:.16f}\n{-d_lon:.16f}\n")
        count += 1

    return count


if __name__ == "__main__":
    # Ensure output directories exist
    for d in [INPUT_DIR, OUTPUT_DIR, LABEL_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    clean(INPUT_DIR)
    clean(OUTPUT_DIR)
    clean(LABEL_DIR)

    filename = Path(__file__).parent.parent / "images" / "pano" / "info.csv"
    pairs = process_data(filename)
    log.info("Training pairs generated: %d", pairs)
    log.info("Output: %s", INPUT_DIR.parent)
