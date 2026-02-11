"""Lightweight cellularity / interest scoring.

This module intentionally avoids any learning-time dependencies.
It provides two fast, classical proxies you can precompute once and
store into the dataset's "density" field (3rd column) to drive
cellularity-guided sampling.

Scores are in [0, 1] and roughly represent the fraction of "cell-like"
foreground within the tissue region of a patch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np

try:
    import cv2  # type: ignore
except Exception as e:  # pragma: no cover
    cv2 = None


@dataclass
class InterestConfig:
    method: Literal["A", "B"] = "A"
    gamma: float = 0.8
    # Tissue mask thresholds (work reasonably for bright backgrounds)
    hsv_s_min: int = 20  # 0..255
    hsv_v_max_bg: int = 245  # consider very bright pixels as background
    min_tissue_ratio: float = 0.10
    # Morphology
    morph_kernel: int = 3
    morph_iter: int = 1
    # Subsampling for method B (KMeans on ab)
    max_pixels_kmeans: int = 20000


def _ensure_cv2():
    if cv2 is None:
        raise ImportError("opencv-python (cv2) is required for interest_score.py")


def _gamma_u8(x_u8: np.ndarray, gamma: float) -> np.ndarray:
    # x_u8: uint8 image channel
    if gamma <= 0:
        return x_u8
    x = x_u8.astype(np.float32) / 255.0
    y = np.power(x, gamma)
    return np.clip(y * 255.0, 0, 255).astype(np.uint8)


def _tissue_mask_hsv(rgb_u8: np.ndarray, cfg: InterestConfig) -> np.ndarray:
    """Return boolean tissue mask."""
    _ensure_cv2()
    hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV)
    s = hsv[..., 1]
    v = hsv[..., 2]
    # tissue if sufficiently saturated OR not extremely bright
    tissue = (s >= cfg.hsv_s_min) | (v <= cfg.hsv_v_max_bg)
    return tissue


def _otsu_threshold(values_u8: np.ndarray) -> int:
    _ensure_cv2()
    if values_u8.size == 0:
        return 255
    # OpenCV expects a 2D uint8 array
    vals = values_u8.reshape(-1, 1)
    thr, _ = cv2.threshold(vals, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return int(thr)


def _morph_cleanup(mask: np.ndarray, cfg: InterestConfig) -> np.ndarray:
    _ensure_cv2()
    k = max(1, int(cfg.morph_kernel))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    m = mask.astype(np.uint8) * 255
    for _ in range(max(1, int(cfg.morph_iter))):
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
    return m > 0


def interest_score(rgb_u8: np.ndarray, cfg: Optional[InterestConfig] = None) -> float:
    """Compute interest/cellularity score for a patch.

    Args:
        rgb_u8: HxWx3 uint8 RGB image.
        cfg: configuration.

    Returns:
        score in [0, 1].
    """
    _ensure_cv2()
    if cfg is None:
        cfg = InterestConfig()

    if rgb_u8.dtype != np.uint8:
        rgb_u8 = np.clip(rgb_u8, 0, 255).astype(np.uint8)

    tissue = _tissue_mask_hsv(rgb_u8, cfg)
    tissue_area = int(tissue.sum())
    if tissue_area <= 0:
        return 0.0
    if tissue_area / float(tissue.size) < float(cfg.min_tissue_ratio):
        return 0.0

    if cfg.method == "A":
        # Method A: "purple-ness" map + Otsu
        # Gamma-correct the V channel to stabilize contrast (cheap illumination fix)
        hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV)
        hsv[..., 2] = _gamma_u8(hsv[..., 2], cfg.gamma)
        rgb2 = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        r = rgb2[..., 0].astype(np.int16)
        g = rgb2[..., 1].astype(np.int16)
        b = rgb2[..., 2].astype(np.int16)
        purple = ((r + b) // 2 - g)
        purple = np.clip(purple + 128, 0, 255).astype(np.uint8)  # shift to uint8

        vals = purple[tissue]
        thr = _otsu_threshold(vals)
        fg = (purple >= thr) & tissue
        fg = _morph_cleanup(fg, cfg)
        fg_area = int(fg.sum())
        return float(fg_area) / float(tissue_area)

    # Method B: LAB (a,b) clustering to adapt to stain variability
    lab = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2LAB)
    a = lab[..., 1]
    b = lab[..., 2]
    ab = np.stack([a, b], axis=-1)
    pts = ab[tissue].reshape(-1, 2).astype(np.float32)
    if pts.shape[0] == 0:
        return 0.0
    if pts.shape[0] > cfg.max_pixels_kmeans:
        idx = np.random.choice(pts.shape[0], cfg.max_pixels_kmeans, replace=False)
        pts_fit = pts[idx]
    else:
        pts_fit = pts

    # 2-means: choose cluster with larger mean 'a' (more magenta-ish)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    _compact, labels, centers = cv2.kmeans(
        pts_fit, 2, None, criteria, 3, cv2.KMEANS_PP_CENTERS
    )
    centers = centers.astype(np.float32)
    target_cluster = int(np.argmax(centers[:, 0]))

    # Assign all tissue pixels to nearest center
    # (avoid running kmeans on all pixels for speed)
    d0 = np.sum((pts - centers[0]) ** 2, axis=1)
    d1 = np.sum((pts - centers[1]) ** 2, axis=1)
    lbl_all = (d1 < d0).astype(np.int32)
    fg_flat = (lbl_all == target_cluster)
    fg = np.zeros(tissue.shape, dtype=bool)
    fg[tissue] = fg_flat
    fg = _morph_cleanup(fg, cfg)
    fg_area = int(fg.sum())
    return float(fg_area) / float(tissue_area)
