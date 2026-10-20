"""Score plate models on a held-out test set in Parda's benchmark format.

Two views of the same run:

* Privacy view (what Parda cares about): run the real pipeline with faces off,
  then ask how much of each labelled plate ended up masked. A plate counts as
  protected at 80% or 95% coverage. Over-mask is the share of the frame masked
  outside any labelled plate.
* Detection view (what detector papers report): recall and precision of the
  raw detections at IoU 0.5, at the pipeline's score threshold.

    python train/eval_plates.py data/plates/parda_test \
        --model plate-yolov9t-384 --model runs/plates/indian/weights/best.onnx
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from parda import models
from parda.bench import load_dataset, score
from parda.detect import PlateDetector, nms
from parda.pipeline import Anonymizer, Config


class UnionDetector:
    """Run several plate models and keep the union of what they find."""

    def __init__(self, paths, thresh, tiles):
        self.dets = [PlateDetector(str(models.resolve(p)), thresh, tiled=tiles) for p in paths]

    def detect(self, img):
        found = [d for det in self.dets for d in det.detect(img)]
        if not found:
            return []
        boxes = np.array([d.box for d in found], np.float32)
        keep = nms(boxes, np.array([d.score for d in found], np.float32), 0.5)
        return [found[i] for i in keep]


def iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match(gts, dets, thr=0.5):
    """Greedy one-to-one matching by score. Returns (true positives, n_gt, n_det)."""
    used = set()
    tp = 0
    for d in sorted(dets, key=lambda d: -d.score):
        best, bi = 0.0, -1
        for i, g in enumerate(gts):
            if i in used:
                continue
            v = iou(d.box, g)
            if v > best:
                best, bi = v, i
        if best >= thr:
            used.add(bi)
            tp += 1
    return tp, len(gts), len(dets)


def run(model: str, root: Path, images, thresh: float, tiles: bool, margin: float = 0.12,
        tile_thresh: float | None = None) -> dict:
    """model is a registered name, an .onnx path, or several joined by '+' (union)."""
    cfg = Config(faces=False, plate_model=model.split("+")[0], plate_thresh=thresh, plate_tiles=tiles,
                 plate_margin=margin, plate_tile_thresh=tile_thresh)
    union = UnionDetector(model.split("+"), thresh, tiles) if "+" in model else None
    anon = Anonymizer(cfg, seed=0, plate_detector=union)
    preds, tp, n_gt, n_det = {}, 0, 0, 0
    t0 = time.perf_counter()
    for im in images:
        img = cv2.imread(str(root / im["file"]))
        anon.reset()
        res = anon.process(img, use_memory=False)
        preds[im["file"]] = [list(map(int, b)) for b in res.masked_boxes]
        gts = [g[1] for g in im["gts"] if g[0] == "plate"]
        a, b, c = match(gts, [d for d in res.detections if d.label == "plate"])
        tp, n_gt, n_det = tp + a, n_gt + b, n_det + c
    secs = time.perf_counter() - t0
    r = score(images, preds)
    out = {
        "model": model, "thresh": thresh, "tiles": tiles, "tile_thresh": tile_thresh, "margin": margin, "images": len(images), "plates": n_gt,
        "protected_80": r.by_group["plate"].recall(0.8), "protected_95": r.by_group["plate"].recall(0.95),
        "over_mask": r.over_mask,
        "recall_iou50": tp / n_gt if n_gt else float("nan"),
        "precision_iou50": tp / n_det if n_det else float("nan"),
        "ms_per_image": 1000 * secs / max(1, len(images)),
        "by_size": {k.split("=", 1)[1]: {"n": t.total, "protected_80": t.recall(0.8)}
                    for k, t in r.by_group.items() if k.startswith("plate / size=")},
        "by_source": {k.split("=", 1)[1]: {"n": t.total, "protected_80": t.recall(0.8)}
                      for k, t in r.by_group.items() if k.startswith("plate / img_source=")},
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--model", action="append", required=True, help="registered name or .onnx path")
    ap.add_argument("--thresh", type=float, default=0.2)
    ap.add_argument("--tiles", action="store_true")
    ap.add_argument("--margin", type=float, default=0.12, help="plate mask margin (Parda default 0.12)")
    ap.add_argument("--tile-thresh", type=float, default=None, help="stricter threshold inside tiles")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    images = load_dataset(args.root)
    rows = []
    for m in args.model:
        row = run(m, args.root, images, args.thresh, args.tiles, args.margin, args.tile_thresh)
        rows.append(row)
        print(json.dumps(row, indent=1), flush=True)
    if args.json:
        args.json.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
