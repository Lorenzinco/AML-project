#!/usr/bin/env python3
"""
Evaluation script for Places2 dataset.

- Loads best weights from data/model_weights
- Evaluates on Places2 (test split by default)
- Computes PSNR and SSIM over the inpainted (hole) regions and over the full image
- Saves qualitative examples and a metrics JSON summary

Usage:
    python -m aml_project.eval_places2 --root AML-project/data/places2 --split test --examples 16

Dataset preparation (Kaggle):
    1) Ensure Kaggle CLI is installed and authenticated.
    2) Download:
       kaggle datasets download -d nickj26/places2-mit-dataset -p <target_dir>
    3) Unzip the archive:
       unzip <target_dir>/places2-mit-dataset.zip -d <target_dir>
    4) Organize images into:
       <root>/images/{train|val|test}/*.jpg|*.png
       Optional masks into:
       <root>/masks/{train|val|test}/*_mask.png or same basenames as images
"""

import argparse
import json
import math
import os
import random
from typing import Any, Dict, List, Optional

import torch
from torch import nn
from torchvision.transforms import Compose, PILToTensor, RandomCrop, Resize
from tqdm import tqdm

from aml_project import model as model_mod
from aml_project.places2_dataset import Places2Dataset
from aml_project.preprocess import Processor
from aml_project.save import save_images
from config import Config

# -------------------------
# Metrics: PSNR and SSIM
# -------------------------


def psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    data_range: float = 1.0,
) -> float:
    """
    Compute PSNR in dB between pred and target.
    pred, target: [B, C, H, W], float in [0,1]
    mask: [B, 1, H, W], 1 where hole (inpainted region), 0 elsewhere. If provided, PSNR is computed on the hole region.
    """
    with torch.no_grad():
        if mask is not None:
            m = mask
            if m.dtype not in (
                torch.float32,
                torch.float64,
                torch.float16,
                torch.bfloat16,
            ):
                m = m.float()
            # If the mask has zero support, return NaN to be skipped upstream
            if float(m.sum().item()) == 0.0:
                return float("nan")
            m = m.expand(-1, pred.shape[1], -1, -1)
            diff = (pred - target) * m
            denom = m.sum() * 1.0 + 1e-8
            mse = diff.pow(2).sum() / denom
        else:
            diff = pred - target
            mse = diff.pow(2).mean()
        mse_val = float(mse.item())
        if not math.isfinite(mse_val):
            return float("nan")
        if mse_val <= 0.0:
            return float("inf")
        return 20.0 * math.log10(data_range) - 10.0 * math.log10(mse_val)


def _ssim_box_filter(
    x: torch.Tensor,
    y: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    data_range: float = 1.0,
    K1: float = 0.01,
    K2: float = 0.03,
    win_size: int = 11,
) -> float:
    """
    Simplified single-scale SSIM using a box filter approximation averaged across channels.
    x, y: [B, C, H, W], float in [0,1]
    mask: [B, 1, H, W] optional, evaluates SSIM on masked region if provided.
    """
    device = x.device
    C = x.shape[1]
    win = torch.ones((1, 1, win_size, win_size), device=device) / (win_size * win_size)

    def filter2d(z: torch.Tensor) -> torch.Tensor:
        z = nn.functional.pad(
            z,
            (win_size // 2, win_size // 2, win_size // 2, win_size // 2),
            mode="reflect",
        )
        return nn.functional.conv2d(
            z, win.expand(z.shape[1], 1, -1, -1), groups=z.shape[1]
        )

    with torch.no_grad():
        if mask is not None:
            m = mask
            if m.dtype not in (
                torch.float32,
                torch.float64,
                torch.float16,
                torch.bfloat16,
            ):
                m = m.float()
            # If mask has zero support, return NaN
            if float(m.sum().item()) == 0.0:
                return float("nan")
            m = m.expand(-1, C, -1, -1)
        else:
            m = None

        mu_x = filter2d(x)
        mu_y = filter2d(y)
        sigma_x = filter2d(x * x) - mu_x * mu_x
        sigma_y = filter2d(y * y) - mu_y * mu_y
        sigma_xy = filter2d(x * y) - mu_x * mu_y

        C1 = (K1 * data_range) ** 2
        C2 = (K2 * data_range) ** 2

        num = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        den = (mu_x.pow(2) + mu_y.pow(2) + C1) * (sigma_x + sigma_y + C2)
        ssim_map = num / (den + 1e-12)

        if m is not None:
            denom = m.sum() + 1e-8
            if float(denom.item()) == 0.0:
                return float("nan")
            val = (ssim_map * m).sum() / denom
        else:
            val = ssim_map.mean()
        val_f = float(val.item())
        if not math.isfinite(val_f):
            return float("nan")
        return val_f


def ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    data_range: float = 1.0,
) -> float:
    """
    Multi-channel SSIM average using a box-filter approximation.
    """
    return _ssim_box_filter(pred, target, mask=mask, data_range=data_range)


# -------------------------
# Evaluation routine
# -------------------------


def eval_places2(
    root: str,
    split: str = "test",
    generate_masks: bool = True,
    mask_type: str = "irregular",
    examples: int = 16,
    save_examples_dir: str = "AML-project/data/places2_examples",
    metrics_out: str = "AML-project/data/places2_eval_metrics.json",
    device_str: Optional[str] = None,
    max_samples: int = 200,
) -> Dict[str, Any]:
    """
    Evaluate the inpainting model on Places2.

    - Loads model and weights from data/model_weights
    - Iterates over the dataset
    - Uses dataset masks if present; otherwise generates masks (default: irregular)
    - Composites final output: fill holes with prediction, keep non-hole regions from original
    - Computes PSNR/SSIM over hole region and full image
    - Saves qualitative examples and a metrics JSON

    Returns:
        metrics dict
    """
    config = Config()
    if device_str is not None:
        config.device = device_str
    device = torch.device(
        config.detect_device() if config.device is None else config.device
    )

    preprocessor = Processor(config)
    # Dataset
    ds = Places2Dataset(
        root=root,
        split=split,
        resolution=config.resolution,
        generate_masks=generate_masks,
        mask_type=mask_type if mask_type in ("center", "irregular") else "irregular",
        external_images_dir="/Users/lorenzinco/Downloads/places_test/test_256",
    )
    total_samples = min(len(ds), max_samples)

    # Model
    net = model_mod.Photosciop(config).to(device)
    # Prefer project-relative path first
    weights_candidates = [
        os.path.join("AML-project", "data", "model_weights"),
        os.path.join("data", "model_weights"),
    ]
    weights_path = None
    for cand in weights_candidates:
        if os.path.exists(cand):
            weights_path = cand
            break
    if weights_path is None:
        raise FileNotFoundError(
            "Could not locate best weights file at AML-project/data/model_weights or data/model_weights"
        )
    state = torch.load(weights_path, map_location=device)
    net.load_state_dict(state)
    net.eval()

    os.makedirs(save_examples_dir, exist_ok=True)

    psnr_hole_vals: List[float] = []
    ssim_hole_vals: List[float] = []
    psnr_full_vals: List[float] = []
    ssim_full_vals: List[float] = []

    # Buckets for hole coverage percentages: 10–20%, 20–30%, 30–40%, 40–50%
    buckets = {
        (0.10, 0.20): {
            "psnr_hole": [],
            "ssim_hole": [],
            "psnr_full": [],
            "ssim_full": [],
            "count": 0,
        },
        (0.20, 0.30): {
            "psnr_hole": [],
            "ssim_hole": [],
            "psnr_full": [],
            "ssim_full": [],
            "count": 0,
        },
        (0.30, 0.40): {
            "psnr_hole": [],
            "ssim_hole": [],
            "psnr_full": [],
            "ssim_full": [],
            "count": 0,
        },
        (0.40, 0.50): {
            "psnr_hole": [],
            "ssim_hole": [],
            "psnr_full": [],
            "ssim_full": [],
            "count": 0,
        },
    }

    example_count = 0

    for i in tqdm(range(total_samples), desc=f"Evaluating Places2 {split}"):
        sample = ds[i]
        img_uint8 = sample["image"]  # [3,H,W], uint8
        img = preprocessor(img_uint8.unsqueeze(0)).to(device)  # [1,3,H,W], normalized

        # Generate mask to evenly target hole coverage buckets per sample (mask==1 known region)
        H, W = int(img_uint8.shape[1]), int(img_uint8.shape[2])
        bucket_ranges = [(0.10, 0.20), (0.20, 0.30), (0.30, 0.40), (0.40, 0.50)]
        lo, hi = bucket_ranges[i % len(bucket_ranges)]
        # Aim for at least the lower bound of the bucket to guarantee coverage >= 10%, 20%, 30%, 40%
        target_ratio = hi
        target_area = int(target_ratio * H * W)

        # Build hole via random walk of adjacent squares with a random start and random directions/rotations until target coverage is met.
        hole = torch.zeros((1, H, W), dtype=torch.uint8)

        # Set base block size (adjacent squares)
        # Increase base block size to produce larger masks (15% of the min image side)
        block_side = max(1, int(0.30 * min(H, W)))

        # Random start position, ensure within bounds
        cur_row = random.randint(0, max(0, H - block_side))
        cur_col = random.randint(0, max(0, W - block_side))

        filled_area = 0
        tries = 0
        max_tries = 10000  # safety to prevent infinite loops

        # Directions including diagonals; rotation is implicit via direction choice
        directions = [
            (-1, 0),  # up
            (1, 0),  # down
            (0, -1),  # left
            (0, 1),  # right
            (-1, -1),  # up-left
            (-1, 1),  # up-right
            (1, -1),  # down-left
            (1, 1),  # down-right
        ]

        while filled_area < target_area and tries < max_tries:
            tries += 1
            # Place current square
            r0 = cur_row
            c0 = cur_col
            r1 = min(r0 + block_side, H)
            c1 = min(c0 + block_side, W)
            # Mark square (adjacent placement, no sparsity)
            hole[:, r0:r1, c0:c1] = 1
            filled_area += (r1 - r0) * (c1 - c0)
            if filled_area >= target_area:
                break

            # Pick a random direction step; ensure adjacency (step size == block_side)
            dr, dc = random.choice(directions)
            next_row = cur_row + dr * block_side
            next_col = cur_col + dc * block_side

            # Clamp to bounds; if out of bounds, try a different random direction
            if not (
                0 <= next_row <= H - block_side and 0 <= next_col <= W - block_side
            ):
                # Try up to a few alternate directions
                alt_dirs = directions[:]
                random.shuffle(alt_dirs)
                moved = False
                for adr, adc in alt_dirs:
                    nr = cur_row + adr * block_side
                    nc = cur_col + adc * block_side
                    if 0 <= nr <= H - block_side and 0 <= nc <= W - block_side:
                        next_row, next_col = nr, nc
                        moved = True
                        break
                if not moved:
                    # If stuck, randomly re-seed near the current cluster to remain adjacent
                    next_row = max(
                        0,
                        min(
                            H - block_side,
                            cur_row + random.randint(-block_side, block_side),
                        ),
                    )
                    next_col = max(
                        0,
                        min(
                            W - block_side,
                            cur_col + random.randint(-block_side, block_side),
                        ),
                    )

            # Advance to next adjacent square
            cur_row, cur_col = next_row, next_col

        # Ensure we reach the target coverage by adding a final adjacent patch trimmed to residual area
        residual = max(0, target_area - filled_area)
        if residual > 0:
            # Try to place a final patch adjacent to the last position, trimming to residual area
            # Compute patch dimensions to match residual area as closely as possible
            # Prefer square-like patch but allow rectangular trim
            patch_h = min(block_side, H - cur_row)
            patch_w_needed = max(1, int(residual / max(1, patch_h)))
            patch_w = min(patch_w_needed, block_side, W - cur_col)
            if patch_w * patch_h < residual:
                # If still under target, try expanding height within bounds
                expand_h = min(block_side, H - cur_row)
                if patch_w * expand_h >= residual:
                    patch_h = expand_h
                else:
                    # If cannot reach exactly, just fill the max available at current position
                    patch_h = expand_h
                    patch_w = min(block_side, W - cur_col)
            r1 = min(cur_row + patch_h, H)
            c1 = min(cur_col + patch_w, W)
            hole[:, cur_row:r1, cur_col:c1] = 1

        # Known region is the complement of the hole; mask==1 denotes known region per project convention
        known = (1 - hole).float().to(device)  # [1,H,W], 1 == known region
        mask = known.unsqueeze(0)  # [1,1,H,W]

        # Ensure hole coverage meets at least the lower bound
        hole_ratio = float((1.0 - mask).mean().item())
        if hole_ratio < lo:
            # Expand the cluster by increasing block size and adding adjacent patches until we reach the lower bound
            expand_side = max(block_side, int(0.35 * min(H, W)))
            cur_row = min(cur_row, max(0, H - expand_side))
            cur_col = min(cur_col, max(0, W - expand_side))
            r1 = min(cur_row + expand_side, H)
            c1 = min(cur_col + expand_side, W)
            hole[:, cur_row:r1, cur_col:c1] = 1
            known = (1 - hole).float().to(device)
            mask = known.unsqueeze(0)

        # Build model input: concatenate masked image and mask (mask==1 known region, match mask_batch)
        masked_img = img * mask
        model_input = torch.cat([masked_img, mask], dim=1)  # [1,4,H,W]

        with torch.no_grad():
            pred = net(model_input)  # [1,3,H,W], normalized
            # Composite: fill holes with prediction (1 - mask), keep known region (mask) from original
            final = pred * (1.0 - mask) + img * mask

        # Metrics on normalized tensors (in [0,1] after denorm)
        ps_hole = psnr(final, img, mask=(1.0 - mask), data_range=1.0)
        ss_hole = ssim(final, img, mask=(1.0 - mask), data_range=1.0)
        ps_full = psnr(final, img, mask=None, data_range=1.0)
        ss_full = ssim(final, img, mask=None, data_range=1.0)

        psnr_hole_vals.append(ps_hole)
        ssim_hole_vals.append(ss_hole)
        psnr_full_vals.append(ps_full)
        ssim_full_vals.append(ss_full)

        # Compute hole coverage ratio (percentage of pixels covered by the hole)
        hole_ratio = float((1.0 - mask).mean().item())  # in [0,1]
        # Assign to buckets based on percentage
        for (lo, hi), store in buckets.items():
            if hole_ratio >= lo and hole_ratio < hi:
                store["psnr_hole"].append(ps_hole)
                store["ssim_hole"].append(ss_hole)
                store["psnr_full"].append(ps_full)
                store["ssim_full"].append(ss_full)
                store["count"] += 1
                break

        # Save qualitative examples
        if example_count < examples:
            masked_input_vis = preprocessor.denorm(masked_img)[0]
            pred_vis = preprocessor.denorm(pred)[0]
            final_vis = preprocessor.denorm(final)[0]
            target_vis = preprocessor.denorm(img)[0]
            to_save = [masked_input_vis, pred_vis, final_vis, target_vis]
            labels = ["input(masked)", "pred", "final(composite)", "target"]
            out_path = os.path.join(save_examples_dir, f"example_{i:05d}.png")
            save_images(to_save, labels, path=out_path, dpi=120)
            example_count += 1

    # Aggregate metrics
    def _agg(vals: List[float]) -> Dict[str, float]:
        if len(vals) == 0:
            return {
                "mean": float("nan"),
                "std": float("nan"),
                "min": float("nan"),
                "max": float("nan"),
            }
        arr = torch.tensor(vals)
        return {
            "mean": float(arr.mean().item()),
            "std": float(arr.std(unbiased=False).item()),
            "min": float(arr.min().item()),
            "max": float(arr.max().item()),
        }

    # Aggregate bucket metrics
    bucket_metrics = {}
    for (lo, hi), store in buckets.items():
        key = f"{int(lo * 100)}-{int(hi * 100)}%"
        bucket_metrics[key] = {
            "count": store["count"],
            "psnr_hole": _agg(store["psnr_hole"]),
            "ssim_hole": _agg(store["ssim_hole"]),
            "psnr_full": _agg(store["psnr_full"]),
            "ssim_full": _agg(store["ssim_full"]),
        }

    metrics = {
        "count": len(ds),
        "psnr_hole": _agg(psnr_hole_vals),
        "ssim_hole": _agg(ssim_hole_vals),
        "psnr_full": _agg(psnr_full_vals),
        "ssim_full": _agg(ssim_full_vals),
        "bucket_metrics": bucket_metrics,
        "config": Config().dict(),
        "split": split,
        "root": root,
        "examples_saved": example_count,
        "weights_path": weights_path,
    }

    os.makedirs(os.path.dirname(metrics_out), exist_ok=True)
    with open(metrics_out, "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate inpainting model on Places2."
    )
    parser.add_argument(
        "--root",
        type=str,
        default="AML-project/data/places2",
        help="Dataset root directory (expects images/{split} and optional masks/{split}); images will be taken from /Users/lorenzinco/Downloads/places_test/test_256 if present",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate",
    )
    parser.add_argument(
        "--generate-masks",
        action="store_true",
        help="Generate masks if dataset masks are absent",
    )
    parser.add_argument(
        "--no-generate-masks",
        dest="generate_masks",
        action="store_false",
        help="Disable mask generation",
    )
    parser.set_defaults(generate_masks=True)
    parser.add_argument(
        "--mask-type",
        type=str,
        default="irregular",
        choices=["center", "irregular"],
        help="Mask type when generating masks",
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=16,
        help="Number of qualitative examples to save",
    )
    parser.add_argument(
        "--examples-dir",
        type=str,
        default="AML-project/data/places2_examples",
        help="Directory to save qualitative examples",
    )
    parser.add_argument(
        "--metrics-out",
        type=str,
        default="AML-project/data/places2_eval_metrics.json",
        help="Path to save metrics JSON",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=200,
        help="Limit evaluation to this many images (default: 200)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Force device (e.g. cpu/cuda/mps)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    metrics = eval_places2(
        root=args.root,
        split=args.split,
        generate_masks=args.generate_masks,
        mask_type=args.mask_type,
        examples=args.examples,
        save_examples_dir=args.examples_dir,
        metrics_out=args.metrics_out,
        device_str=args.device,
        max_samples=args.max_samples,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
