#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
街景全景图黑边检测与裁剪工具
检测并裁剪图片底部的黑边，按 2:1 比例从左上角裁剪
纯 PIL + numpy 实现，无 opencv 依赖
"""

from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image


# ── Constants ────────────────────────────────────────────────────────────────

TARGET_ASPECT_RATIO = 2.0  # 宽高比 2:1
BOTTOM_BLACK_EDGE_RATIO = 0.05  # 底部黑边检测阈值（占图片高度的比例）


# ── In-memory helpers ────────────────────────────────────────────────────────

def crop_black_edge_from_image(img: Image.Image, threshold: int = 15) -> Image.Image:
    """
    检测并裁剪图片底部黑边，纯 in-memory 操作（不落盘）。

    Args:
        img:       PIL Image 对象
        threshold: 黑边检测阈值 0-255

    Returns:
        裁剪后的 PIL Image
    """
    image = np.array(img.convert("RGB"))
    h, w = image.shape[:2]
    gray = np.mean(image, axis=2)

    # 从底部向上扫描，找到第一个非黑像素行
    valid_bottom = h
    for i in range(h - 1, -1, -1):
        if gray[i, :].mean() > threshold:
            valid_bottom = i + 1
            break

    bottom_black_height = h - valid_bottom
    if bottom_black_height > h * BOTTOM_BLACK_EDGE_RATIO:
        image = image[:valid_bottom, :]

    return Image.fromarray(image)


# ── Core functions ───────────────────────────────────────────────────────────

def detect_and_crop_black_edge(
    input_path: Path | str,
    output_path: Path | str,
    threshold: int = 15,
) -> bool:
    """
    检测并裁剪单张图片的底部黑边。

    Args:
        input_path:  输入图片路径
        output_path: 输出图片路径
        threshold:   黑边检测阈值 0-255（像素平均值低于此值视为黑）

    Returns:
        bool: True 表示裁剪了黑边，False 表示无黑边（图片正常）
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    img = Image.open(str(input_path))
    img = img.convert("RGB")
    image = np.array(img)
    h, w = image.shape[:2]

    # 转换为灰度图 (H, W)
    gray = np.mean(image, axis=2).astype(np.uint8)

    # 从底部向上扫描，找到第一个非黑像素行
    valid_bottom = h
    for i in range(h - 1, -1, -1):
        if gray[i, :].mean() > threshold:
            valid_bottom = i + 1
            break

    bottom_black_height = h - valid_bottom
    has_bottom_black_border = bottom_black_height > h * BOTTOM_BLACK_EDGE_RATIO

    if not has_bottom_black_border:
        # 无黑边，复制原图
        img.save(str(output_path))
        return False

    # 去除底部黑边
    valid_image = image[:valid_bottom, :]
    valid_h, valid_w = valid_image.shape[:2]

    # 按 2:1 比例从左上角裁剪
    ideal_width = int(valid_h * TARGET_ASPECT_RATIO)
    if ideal_width <= valid_w:
        cropped = valid_image[:, :ideal_width]
    else:
        ideal_height = int(valid_w / TARGET_ASPECT_RATIO)
        cropped = valid_image[:ideal_height, :]

    # 拉伸回原始分辨率
    final_img = Image.fromarray(cropped)
    final_img = final_img.resize((w, h), Image.BICUBIC)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_img.save(str(output_path))
    return True


def process_directory(
    input_dir: Path | str,
    output_dir: Path | str,
    threshold: int = 15,
) -> dict:
    """
    批量处理目录下所有图片的黑边检测与裁剪。

    Args:
        input_dir:  输入图片目录
        output_dir: 输出图片目录
        threshold:  黑边检测阈值

    Returns:
        dict: 处理统计 {'total', 'cropped', 'unchanged', 'failed'}
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    image_files = []
    for ext in image_extensions:
        image_files.extend(input_dir.glob(f"*{ext}"))
        image_files.extend(input_dir.glob(f"*{ext.upper()}"))

    stats = {"total": len(image_files), "cropped": 0, "unchanged": 0, "failed": 0}

    for img_path in image_files:
        out_path = output_dir / img_path.name
        try:
            cropped = detect_and_crop_black_edge(img_path, out_path, threshold)
            if cropped:
                stats["cropped"] += 1
            else:
                stats["unchanged"] += 1
        except Exception:
            stats["failed"] += 1

    return stats
