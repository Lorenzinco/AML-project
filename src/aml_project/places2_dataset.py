"""
Places2 dataset loader with optional mask generation and preparation helpers.

This module provides:
- A PyTorch Dataset class (`Places2Dataset`) to load Places2 images (and masks if present).
- Optional synthetic mask generation (center or irregular) when masks are not available.
- Helper functions to prepare the directory structure and guidance for downloading from Kaggle.

Dataset reference:
- Kaggle: https://www.kaggle.com/datasets/nickj26/places2-mit-dataset

Expected directory layout after preparation:
root/
  images/
    train/
      *.jpg|*.png
    val/
      *.jpg|*.png
    test/
      *.jpg|*.png
  masks/ (optional)
    train/
      *_mask.png (or same basename as image)
    val/
      *_mask.png
    test/
      *_mask.png

If your layout differs, you can pass custom subpaths to `Places2Dataset`.

Note on downloading:
- You can use the Kaggle CLI to download and extract the dataset:
    kaggle datasets download -d nickj26/places2-mit-dataset -p <target_dir>
    unzip <target_dir>/places2-mit-dataset.zip -d <target_dir>
- Then organize the images under the expected layout shown above.

Usage example:
    from aml_project.places2_dataset import Places2Dataset
    from config import Config
    cfg = Config()
    ds = Places2Dataset(
        root="AML-project/data/places2",
        split="test",
        resolution=cfg.resolution,
        generate_masks=True,
        mask_type="irregular",
    )
    sample = ds[0]
    img_uint8 = sample["image"]   # [3,H,W] uint8
    mask_u8 = sample["mask"]      # [1,H,W] uint8 {0,1}
    path = sample["path"]

"""

from __future__ import annotations

import os
from typing import Dict, List, Literal, Optional

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision.transforms import Compose, PILToTensor, Resize

try:
    # Optional: use your project Config to align resolution/dtype; otherwise use defaults.
    from config import Config  # type: ignore
except Exception:
    Config = None  # type: ignore


# ------------------------------
# Utilities: filesystem and I/O
# ------------------------------


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _is_image_file(name: str) -> bool:
    low = name.lower()
    return low.endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))


def _default_resolution() -> tuple[int, int]:
    if Config is not None:
        try:
            return Config().resolution
        except Exception:
            pass
    return (128, 128)


def _to_tensor_rgb(img: Image.Image, resolution: tuple[int, int]) -> torch.Tensor:
    """
    Convert PIL image to Tensor [C,H,W], resize to resolution.
    Does not normalize; use your project's preprocessor for normalization.
    """
    x = PILToTensor()(img.convert("RGB"))
    x = Resize(resolution)(x)
    return x


def _load_mask_file(path: str, resolution: tuple[int, int]) -> Optional[torch.Tensor]:
    """
    Load a mask image (assumed binary or grayscale) and return [1,H,W] uint8 tensor with values {0,1}.
    Returns None if path doesn't exist.
    """
    if not os.path.exists(path):
        return None
    with Image.open(path) as im:
        im = ImageOps.grayscale(im)
        arr = np.array(im, dtype=np.uint8)
        # Threshold to binary at 128
        bin_arr = (arr >= 128).astype(np.uint8)
        t = torch.from_numpy(bin_arr)  # [H,W], uint8
        t = Resize(resolution)(t.unsqueeze(0))  # [1,H,W]
        t = (t > 0.5).to(torch.uint8)
        return t


# -----------------------
# Synthetic mask generator
# -----------------------


def _generate_center_mask(
    height: int,
    width: int,
    ratio: float = 0.25,
) -> torch.Tensor:
    """
    Generate a center square mask of given area ratio.
    mask==1 denotes the hole region; mask==0 denotes known pixels.
    Returns [1,H,W] uint8.
    """
    h, w = height, width
    side = int(np.sqrt(ratio) * min(h, w))
    side = max(1, min(side, min(h, w)))
    top = (h - side) // 2
    left = (w - side) // 2
    mask = torch.zeros((1, h, w), dtype=torch.uint8)
    mask[:, top : top + side, left : left + side] = 1
    return mask


def _draw_ellipse(mask: torch.Tensor, cy: int, cx: int, ry: int, rx: int) -> None:
    """
    Rasterize a filled ellipse onto mask (in-place). mask: [1,H,W], uint8.
    """
    h, w = mask.shape[-2], mask.shape[-1]
    y, x = np.ogrid[:h, :w]
    ellipse = (((y - cy) / max(ry, 1)) ** 2 + ((x - cx) / max(rx, 1)) ** 2) <= 1.0
    mask[0, ellipse] = 1


def _draw_line(
    mask: torch.Tensor, y0: int, x0: int, y1: int, x1: int, thickness: int = 3
) -> None:
    """
    Rasterize a line with thickness onto mask (in-place). mask: [1,H,W], uint8.
    Uses a simple Bresenham-like approach with dilation.
    """
    h, w = mask.shape[-2], mask.shape[-1]
    y0, x0, y1, x1 = int(y0), int(x0), int(y1), int(x1)
    dy = abs(y1 - y0)
    dx = abs(x1 - x0)
    sy = 1 if y0 < y1 else -1
    sx = 1 if x0 < x1 else -1
    err = dx - dy

    y, x = y0, x0
    while True:
        y_min = max(0, y - thickness)
        y_max = min(h, y + thickness + 1)
        x_min = max(0, x - thickness)
        x_max = min(w, x + thickness + 1)
        mask[0, y_min:y_max, x_min:x_max] = 1

        if y == y1 and x == x1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy


def _generate_irregular_mask(
    height: int,
    width: int,
    num_ellipses: tuple[int, int] = (3, 10),
    num_lines: tuple[int, int] = (6, 15),
    ellipse_size_range: tuple[float, float] = (0.05, 0.2),
    line_thickness_range: tuple[int, int] = (3, 7),
    rng: Optional[np.random.Generator] = None,
) -> torch.Tensor:
    """
    Generate an irregular mask composed of random ellipses and lines.
    Returns [1,H,W] uint8 with values {0,1}, where 1 denotes the hole region.
    """
    if rng is None:
        rng = np.random.default_rng()

    mask = torch.zeros((1, height, width), dtype=torch.uint8)

    # Ellipses
    ne = rng.integers(num_ellipses[0], num_ellipses[1] + 1)
    for _ in range(int(ne)):
        cy = int(rng.integers(0, height))
        cx = int(rng.integers(0, width))
        base = int(
            min(height, width)
            * rng.uniform(ellipse_size_range[0], ellipse_size_range[1])
        )
        ry = int(base * rng.uniform(0.5, 1.5))
        rx = int(base * rng.uniform(0.5, 1.5))
        _draw_ellipse(mask, cy, cx, max(1, ry), max(1, rx))

    # Lines
    nl = rng.integers(num_lines[0], num_lines[1] + 1)
    for _ in range(int(nl)):
        y0 = int(rng.integers(0, height))
        x0 = int(rng.integers(0, width))
        y1 = int(rng.integers(0, height))
        x1 = int(rng.integers(0, width))
        thickness = int(
            rng.integers(line_thickness_range[0], line_thickness_range[1] + 1)
        )
        _draw_line(mask, y0, x0, y1, x1, max(1, thickness))

    mask = (mask > 0).to(torch.uint8)
    return mask


# ------------------------
# Dataset implementation
# ------------------------


class Places2Dataset(Dataset):
    """
    PyTorch dataset for Places2 images with optional masks.

    If masks are present:
      - tries to find matching mask files by:
        1) images/{split}/image.ext -> masks/{split}/image_mask.ext
        2) images/{split}/image.ext -> masks/{split}/image.ext
      - any found mask is loaded and resized to match the target resolution

    If masks are not present:
      - you may choose to generate masks on-the-fly:
        generate_masks=True with mask_type in {"center", "irregular"}

    Returns:
      dict with:
        {
          "image": uint8 Tensor [3,H,W] (0..255),
          "mask":  uint8 Tensor [1,H,W] with values {0,1} (1 denotes hole),
          "path":  str (image file path)
        }

    Use your existing `aml_project.preprocess.Processor` to normalize images for the model.
    """

    def __init__(
        self,
        root: str,
        split: Literal["train", "val", "test"] = "test",
        images_subdir: str = "images",
        masks_subdir: str = "masks",
        resolution: Optional[tuple[int, int]] = None,
        generate_masks: bool = True,
        mask_type: Literal["center", "irregular"] = "irregular",
        center_ratio: float = 0.25,
        num_ellipses: tuple[int, int] = (3, 10),
        num_lines: tuple[int, int] = (6, 15),
        ellipse_size_range: tuple[float, float] = (0.05, 0.2),
        line_thickness_range: tuple[int, int] = (3, 7),
        rng: Optional[np.random.Generator] = None,
        external_images_dir: Optional[str] = None,
        target_hole_range: Optional[tuple[float, float]] = None,
        max_hole_tries: int = 10,
    ):
        super().__init__()
        self.root = root
        self.split = split
        # If an external images directory is provided, use it directly (ignores split/images_subdir)
        self.images_dir = (
            external_images_dir
            if external_images_dir
            else os.path.join(root, images_subdir, split)
        )
        self.masks_dir = os.path.join(root, masks_subdir, split)
        self.resolution = resolution or _default_resolution()
        self.generate_masks = generate_masks
        self.mask_type = mask_type
        self.center_ratio = center_ratio
        self.num_ellipses = num_ellipses
        self.num_lines = num_lines
        self.ellipse_size_range = ellipse_size_range
        self.line_thickness_range = line_thickness_range
        self.rng = rng or np.random.default_rng()

        if not os.path.isdir(self.images_dir):
            raise RuntimeError(
                f"Images directory not found: {self.images_dir}. "
                f"Provide a valid external_images_dir or prepare the dataset under {os.path.join(root, images_subdir)}."
            )

        self.image_paths: List[str] = []
        for fname in sorted(os.listdir(self.images_dir)):
            if _is_image_file(fname):
                self.image_paths.append(os.path.join(self.images_dir, fname))

        if len(self.image_paths) == 0:
            raise RuntimeError(
                f"No image files found in {self.images_dir}. "
                "Please ensure the dataset is correctly placed."
            )

        self.masks_available = (
            os.path.isdir(self.masks_dir) and len(os.listdir(self.masks_dir)) > 0
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def _find_mask_for_image(self, image_path: str) -> Optional[str]:
        if not self.masks_available:
            return None
        base = os.path.basename(image_path)
        stem, ext = os.path.splitext(base)
        candidates = [
            os.path.join(self.masks_dir, stem + "_mask" + ext),
            os.path.join(self.masks_dir, stem + ext),
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
        return None

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        img_path = self.image_paths[idx]
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            image_tensor = _to_tensor_rgb(im, self.resolution)  # [3,H,W], uint8

        mask_tensor: Optional[torch.Tensor] = None

        # Try dataset-provided mask first
        mask_path = self._find_mask_for_image(img_path)
        if mask_path is not None:
            mask_tensor = _load_mask_file(mask_path, self.resolution)

        # If no dataset mask, optionally generate one
        if mask_tensor is None and self.generate_masks:
            H, W = self.resolution
            if self.mask_type == "center":
                # Generate center mask and, if a target coverage is provided, adjust ratio to fit
                ratio = self.center_ratio
                if hasattr(self, "target_hole_range") and self.target_hole_range:
                    # Try to fit target hole coverage by adjusting ratio
                    lo, hi = self.target_hole_range
                    for _ in range(getattr(self, "max_hole_tries", 10)):
                        mask_candidate = _generate_center_mask(H, W, ratio=ratio)
                        hole_ratio = float((mask_candidate == 1).float().mean().item())
                        if lo <= hole_ratio < hi:
                            mask_tensor = mask_candidate
                            break
                        # Adjust ratio heuristically toward target range midpoint
                        mid = (lo + hi) / 2.0
                        if hole_ratio < lo:
                            ratio = min(0.9, ratio * 1.25)
                        elif hole_ratio >= hi:
                            ratio = max(0.01, ratio * 0.75)
                    if mask_tensor is None:
                        mask_tensor = _generate_center_mask(H, W, ratio=ratio)
                else:
                    mask_tensor = _generate_center_mask(H, W, ratio=ratio)
            elif self.mask_type == "irregular":
                # Generate irregular masks and, if a target coverage is provided, sample until within range
                if hasattr(self, "target_hole_range") and self.target_hole_range:
                    lo, hi = self.target_hole_range
                    mask_candidate = None
                    for _ in range(getattr(self, "max_hole_tries", 10)):
                        mask_candidate = _generate_irregular_mask(
                            H,
                            W,
                            num_ellipses=self.num_ellipses,
                            num_lines=self.num_lines,
                            ellipse_size_range=self.ellipse_size_range,
                            line_thickness_range=self.line_thickness_range,
                            rng=self.rng,
                        )
                        hole_ratio = float((mask_candidate == 1).float().mean().item())
                        if lo <= hole_ratio < hi:
                            mask_tensor = mask_candidate
                            break
                    if mask_tensor is None:
                        # Fallback to last candidate or a fresh one
                        mask_tensor = (
                            mask_candidate
                            if mask_candidate is not None
                            else _generate_irregular_mask(
                                H,
                                W,
                                num_ellipses=self.num_ellipses,
                                num_lines=self.num_lines,
                                ellipse_size_range=self.ellipse_size_range,
                                line_thickness_range=self.line_thickness_range,
                                rng=self.rng,
                            )
                        )
                else:
                    mask_tensor = _generate_irregular_mask(
                        H,
                        W,
                        num_ellipses=self.num_ellipses,
                        num_lines=self.num_lines,
                        ellipse_size_range=self.ellipse_size_range,
                        line_thickness_range=self.line_thickness_range,
                        rng=self.rng,
                    )
            else:
                raise ValueError(f"Unknown mask_type: {self.mask_type}")

        if mask_tensor is None:
            # Provide a zero mask if neither dataset mask nor generation is enabled.
            mask_tensor = torch.zeros(
                (1, self.resolution[0], self.resolution[1]), dtype=torch.uint8
            )

        # Convert to project convention: mask==1 denotes known region, 0 denotes hole
        mask_tensor = 1 - mask_tensor
        return {
            "image": image_tensor,  # uint8 [3,H,W]
            "mask": mask_tensor,  # uint8 [1,H,W], values {0,1}; 1 denotes known region
            "path": img_path,
        }


# -----------------------------------
# Preparation and convenience helpers
# -----------------------------------


def places2_prepare_layout(
    root: str,
    images_subdir: str = "images",
    masks_subdir: str = "masks",
    splits: Optional[List[str]] = None,
) -> None:
    """
    Create the expected directory layout for Places2 dataset under `root`.
    This does NOT download data. It only creates directories:
      root/images/{train,val,test}
      root/masks/{train,val,test}
    """
    if splits is None:
        splits = ["train", "val", "test"]
    for sub in (images_subdir, masks_subdir):
        for sp in splits:
            _ensure_dir(os.path.join(root, sub, sp))


def places2_download_instructions() -> str:
    """
    Return a string with instructions to download Places2 from Kaggle
    and organize it for this loader.
    """
    return (
        "To download Places2 from Kaggle:\n"
        "1) Ensure Kaggle CLI is installed and authenticated (kaggle.json configured).\n"
        "2) Run:\n"
        "   kaggle datasets download -d nickj26/places2-mit-dataset -p <target_dir>\n"
        "3) Unzip the downloaded archive:\n"
        "   unzip <target_dir>/places2-mit-dataset.zip -d <target_dir>\n"
        "4) Organize images into:\n"
        "   <root>/images/train, <root>/images/val, <root>/images/test\n"
        "   and optional masks into:\n"
        "   <root>/masks/train, <root>/masks/val, <root>/masks/test\n"
        "5) Use Places2Dataset with root=<root>.\n"
    )
