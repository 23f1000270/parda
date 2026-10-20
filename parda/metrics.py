"""Evaluation metrics built for anonymisation, not detection.

Detection benchmarks reward tight boxes (IoU). For privacy the question is
different: after masking, how much of each face or plate is still visible?

* coverage(g) = fraction of ground-truth box g that ends up masked.
* An object counts as protected at level t if coverage(g) >= t.
* protected recall@t = protected objects / all objects.
* over-mask = fraction of the frame masked outside any ground-truth box,
  which measures how much picture we sacrifice.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

Box = Sequence[float]


def mask_raster(boxes: Iterable[Box], w: int, h: int) -> np.ndarray:
    m = np.zeros((h, w), dtype=bool)
    for x1, y1, x2, y2 in boxes:
        x1, y1 = max(0, int(np.floor(x1))), max(0, int(np.floor(y1)))
        x2, y2 = min(w, int(np.ceil(x2))), min(h, int(np.ceil(y2)))
        if x2 > x1 and y2 > y1:
            m[y1:y2, x1:x2] = True
    return m


def coverage(gt: Box, raster: np.ndarray) -> float:
    h, w = raster.shape
    x1, y1 = max(0, int(np.floor(gt[0]))), max(0, int(np.floor(gt[1])))
    x2, y2 = min(w, int(np.ceil(gt[2]))), min(h, int(np.ceil(gt[3])))
    if x2 <= x1 or y2 <= y1:
        return 1.0
    return float(raster[y1:y2, x1:x2].mean())


@dataclass
class Tally:
    total: int = 0
    protected: dict = field(default_factory=lambda: defaultdict(int))

    def add(self, cov: float, levels: Sequence[float]) -> None:
        self.total += 1
        for t in levels:
            if cov >= t:
                self.protected[t] += 1

    def recall(self, t: float) -> float:
        return self.protected[t] / self.total if self.total else float("nan")


@dataclass
class EvalResult:
    levels: tuple
    by_group: dict  # group name -> Tally
    over_mask: float
    frames: int


def evaluate(samples, levels=(0.8, 0.95)) -> EvalResult:
    """samples: iterable of (width, height, gt_list, mask_boxes) where gt_list is
    [(label, box, attrs_dict)] and mask_boxes is [box]."""
    groups: dict[str, Tally] = defaultdict(Tally)
    over_total = 0.0
    n = 0
    for w, h, gts, masks in samples:
        n += 1
        r = mask_raster(masks, w, h)
        gt_r = mask_raster([g[1] for g in gts], w, h)
        over_total += float((r & ~gt_r).mean())
        for label, box, attrs in gts:
            c = coverage(box, r)
            groups[f"{label}"].add(c, levels)
            for k, v in (attrs or {}).items():
                groups[f"{label} / {k}={v}"].add(c, levels)
    return EvalResult(tuple(levels), dict(groups), over_total / max(n, 1), n)


def to_markdown(res: EvalResult) -> str:
    head = "| group | objects | " + " | ".join(f"protected @{t:.2f}" for t in res.levels) + " |"
    sep = "|---|---:|" + "---:|" * len(res.levels)
    rows = []
    for name in sorted(res.by_group, key=lambda s: (s.count("/"), s)):
        t = res.by_group[name]
        rows.append(f"| {name} | {t.total} | " + " | ".join(f"{t.recall(l):.3f}" for l in res.levels) + " |")
    tail = f"\nFrames: {res.frames}. Mean over-mask (masked area outside any labelled object): {res.over_mask:.4f}"
    return "\n".join([head, sep, *rows]) + "\n" + tail
