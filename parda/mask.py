"""Irreversible masking.

Plain Gaussian blur and fine pixelation can be partly undone by learned
models (McPherson, Shokri and Shmatikov, "Defeating Image Obfuscation with
Deep Learning", 2016). Parda's default mask therefore keeps only a coarse grid
of averaged colour (at most `blocks` cells across the longer side of each
box) and adds random per-cell noise, so no fine detail survives to be
recovered. A solid fill is also available.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

Box = Sequence[float]


def expand(box: Box, frac: float, w: int, h: int) -> tuple[int, int, int, int]:
    """Grow a box by `frac` of its size on every side and clip to the frame."""
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    x1 -= bw * frac
    x2 += bw * frac
    y1 -= bh * frac
    y2 += bh * frac
    return (
        int(max(0, np.floor(x1))),
        int(max(0, np.floor(y1))),
        int(min(w, np.ceil(x2))),
        int(min(h, np.ceil(y2))),
    )


def _blocks(region: np.ndarray, blocks: int, rng: np.random.Generator, noise: int) -> np.ndarray:
    h, w = region.shape[:2]
    cell = max(1, int(np.ceil(max(h, w) / blocks)))
    out = np.empty_like(region)
    for y in range(0, h, cell):
        for x in range(0, w, cell):
            cellv = region[y:y + cell, x:x + cell]
            mean = cellv.reshape(-1, cellv.shape[-1]).mean(axis=0)
            # Same jitter on every channel: brightness noise, not rainbow speckle.
            jitter = int(rng.integers(-noise, noise + 1)) if noise else 0
            out[y:y + cell, x:x + cell] = np.clip(mean + jitter, 0, 255).astype(region.dtype)
    return out


def apply_masks(
    img: np.ndarray,
    boxes: Iterable[tuple[int, int, int, int]],
    mode: str = "blocks",
    blocks: int = 6,
    noise: int = 14,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Return a copy of img with every box irreversibly masked."""
    rng = rng or np.random.default_rng()
    out = img.copy()
    for x1, y1, x2, y2 in boxes:
        if x2 <= x1 or y2 <= y1:
            continue
        region = out[y1:y2, x1:x2]
        if mode == "solid":
            region[:] = region.reshape(-1, region.shape[-1]).mean(axis=0).astype(region.dtype)
        elif mode == "blocks":
            region[:] = _blocks(region, blocks, rng, noise)
        elif mode == "black":
            region[:] = 0
        else:
            raise ValueError(f"Unknown mask mode '{mode}'")
    return out


def full_frame(img: np.ndarray) -> np.ndarray:
    """Fail-closed mask: hide the whole frame."""
    out = np.zeros_like(img)
    out[:] = 32
    return out
