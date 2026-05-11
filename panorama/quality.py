from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class QualityMetrics:
    bottom_black_ratio: float
    base_sharpness: float
    final_sharpness: float
    sharpness_ratio: float
    heading_shift_px: int
    heading_mean_abs_diff: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def heading_shift_pixels(width: int, heading: float) -> int:
    if width <= 0:
        raise ValueError("width must be positive")
    return int((heading % 360) / 360.0 * width)


def apply_heading_adjustment(image: Image.Image, heading: float) -> Image.Image:
    shift = heading_shift_pixels(image.width, heading)
    if shift == 0:
        return image.copy()
    adjusted = Image.new(image.mode, image.size)
    left = image.crop((0, 0, shift, image.height))
    right = image.crop((shift, 0, image.width, image.height))
    adjusted.paste(right, (0, 0))
    adjusted.paste(left, (image.width - shift, 0))
    return adjusted


def bottom_black_edge_ratio(image: Image.Image, threshold: int = 15) -> float:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    h = arr.shape[0]
    gray = arr.mean(axis=2)
    valid_bottom = 0
    for row in range(h - 1, -1, -1):
        if gray[row, :].mean() > threshold:
            valid_bottom = row + 1
            break
    return float((h - valid_bottom) / h)


def sharpness_score(image: Image.Image) -> float:
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0
    laplacian = (
        gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
        - 4 * gray[1:-1, 1:-1]
    )
    return float(laplacian.var())


def mean_absolute_difference(left: Image.Image, right: Image.Image) -> float:
    if left.size != right.size:
        raise ValueError("images must have the same size")
    left_arr = np.asarray(left.convert("RGB"), dtype=np.float32)
    right_arr = np.asarray(right.convert("RGB"), dtype=np.float32)
    return float(np.abs(left_arr - right_arr).mean())


def build_quality_metrics(
    base_image: Image.Image,
    final_image: Image.Image,
    heading: float,
    threshold: int = 15,
) -> QualityMetrics:
    expected = apply_heading_adjustment(base_image, heading)
    base_sharpness = sharpness_score(base_image)
    final_sharpness = sharpness_score(final_image)
    if base_sharpness <= 1e-9:
        sharpness_ratio = 1.0 if final_sharpness <= 1e-9 else 0.0
    else:
        sharpness_ratio = final_sharpness / base_sharpness
    return QualityMetrics(
        bottom_black_ratio=bottom_black_edge_ratio(final_image, threshold=threshold),
        base_sharpness=base_sharpness,
        final_sharpness=final_sharpness,
        sharpness_ratio=sharpness_ratio,
        heading_shift_px=heading_shift_pixels(final_image.width, heading),
        heading_mean_abs_diff=mean_absolute_difference(final_image, expected),
    )
