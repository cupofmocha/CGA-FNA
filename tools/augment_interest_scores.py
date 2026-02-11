"""Augment existing Cyto-AL dataset .npy with a cellularity/interest score.

This keeps code changes minimal:
  - Cyto-AL already uses the 3rd column ("density") for sampling/filtering.
  - We replace that column with an interest score computed from the patch RGB.

Usage examples:
  python tools/augment_interest_scores.py \
    --in_npy ./data_infor/train_label_new_pred.npy \
    --out_npy ./data_infor/train_label_interestA.npy \
    --image_root ./labeled\ data \
    --method A --gamma 0.8

  python tools/augment_interest_scores.py \
    --in_npy ./data_infor/test_label_new_pred.npy \
    --out_npy ./data_infor/test_label_interestB.npy \
    --image_root ./labeled\ data \
    --method B
"""

from __future__ import annotations

import argparse
import os
from typing import Any

import numpy as np
from PIL import Image

from interest_score import InterestConfig, interest_score


def _resolve_image_path(p: str, image_root: str | None) -> str:
    # Prefer absolute
    if os.path.isabs(p) and os.path.exists(p):
        return p
    # If dataset stores relative path, try image_root
    if image_root is not None:
        cand = os.path.join(image_root, p)
        if os.path.exists(cand):
            return cand
    # Fallback: try relative as-is
    if os.path.exists(p):
        return p
    raise FileNotFoundError(f"Could not resolve image path: {p} (image_root={image_root})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_npy", required=True)
    ap.add_argument("--out_npy", required=True)
    ap.add_argument("--image_root", default=None, help="Root dir to prepend to stored img_path")
    ap.add_argument("--method", choices=["A", "B"], default="A")
    ap.add_argument("--gamma", type=float, default=0.8)
    ap.add_argument("--limit", type=int, default=0, help="Optional limit for quick tests")
    args = ap.parse_args()

    arr = np.load(args.in_npy, allow_pickle=True)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"Unexpected dataset shape: {arr.shape}. Expect (N, >=3).")

    n = arr.shape[0] if args.limit <= 0 else min(arr.shape[0], args.limit)
    out = arr.copy()

    cfg = InterestConfig(method=args.method, gamma=args.gamma)
    scores = np.zeros((n,), dtype=np.float32)

    for i in range(n):
        img_path = str(arr[i, 0])
        p = _resolve_image_path(img_path, args.image_root)
        rgb = np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8)
        scores[i] = interest_score(rgb, cfg)

    # Replace the "density" column (index 2) for the first n rows.
    # (If you used --limit, only a prefix is replaced.)
    out[:n, 2] = scores.astype(object)

    os.makedirs(os.path.dirname(args.out_npy) or ".", exist_ok=True)
    np.save(args.out_npy, out, allow_pickle=True)

    # Also save the raw vector for debugging/plotting.
    vec_path = os.path.splitext(args.out_npy)[0] + "_interest_vec.npy"
    np.save(vec_path, scores)
    print(f"wrote: {args.out_npy}")
    print(f"wrote: {vec_path}  (mean={scores.mean():.4f}, p5={np.percentile(scores,5):.4f}, p95={np.percentile(scores,95):.4f})")


if __name__ == "__main__":
    main()
