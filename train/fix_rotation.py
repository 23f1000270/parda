"""Find and fix images stored sideways relative to their labels.

Some images in the public datasets were saved rotated by 90 or 180 degrees
while their boxes were drawn on the upright picture, so a box lands on empty
road. For every image where a stock plate detector matches none of the
labelled boxes, we try the three other rotations. If a rotation makes the
detector agree with the labels, the pixels are rotated to fit the labels.

The detector is only used to choose between rotations of the same picture.
Images it cannot match in any rotation are left alone and listed for a human
to look at, so the test set is not filtered down to what the stock model
already finds.

    python train/fix_rotation.py data/plates
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from parda import models
from parda.detect import PlateDetector

ROT = {0: None, 90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def iou(a, b) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def matched(img, rel_boxes, det, thr=0.3) -> int:
    h, w = img.shape[:2]
    gts = [((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h) for cx, cy, bw, bh in rel_boxes]
    dets = [d.box for d in det.detect(img)]
    return sum(any(iou(g, d) >= thr for d in dets) for g in gts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--model", default="plate-yolov9t-640")
    args = ap.parse_args()
    det = PlateDetector(str(models.resolve(args.model)), score_thresh=0.1, tiled=True)
    report = {"rotated": [], "unmatched": []}
    for split in ("test", "val", "train"):
        imgs = sorted((args.root / split / "images").glob("*.jpg"))
        for k, p in enumerate(imgs):
            lab = args.root / split / "labels" / (p.stem + ".txt")
            boxes = [tuple(map(float, l.split()[1:5])) for l in lab.read_text().split("\n") if l.strip()]
            img = cv2.imread(str(p))
            if matched(img, boxes, det):
                continue
            best, best_n = 0, 0
            for angle in (90, 270, 180):
                n = matched(cv2.rotate(img, ROT[angle]), boxes, det)
                if n > best_n:
                    best, best_n = angle, n
            if best:
                cv2.imwrite(str(p), cv2.rotate(img, ROT[best]), [cv2.IMWRITE_JPEG_QUALITY, 95])
                report["rotated"].append({"file": f"{split}/images/{p.name}", "angle": best})
            else:
                report["unmatched"].append(f"{split}/images/{p.name}")
            if k % 200 == 0:
                print(split, k, len(imgs), "rotated", len(report["rotated"]), "unmatched", len(report["unmatched"]),
                      flush=True)
    # Keep the Parda copy of the test split in step.
    pt = args.root / "parda_test"
    meta = json.loads((pt / "labels.json").read_text())
    by_file = {im["file"]: im for im in meta["images"]}
    for r in report["rotated"]:
        if r["file"].startswith("test/"):
            name = Path(r["file"]).name
            src = args.root / r["file"]
            (pt / "frames" / name).write_bytes(src.read_bytes())
            im = by_file[f"frames/{name}"]
            h, w = cv2.imread(str(src)).shape[:2]
            im["width"], im["height"] = w, h
            labels = (args.root / "test" / "labels" / (Path(name).stem + ".txt")).read_text().split("\n")
            rel = [tuple(map(float, l.split()[1:5])) for l in labels if l.strip()]
            anns = [a for a in meta["annotations"] if a["image_id"] == im["id"]]
            for a, (cx, cy, bw, bh) in zip(anns, rel):
                a["bbox"] = [round((cx - bw / 2) * w, 1), round((cy - bh / 2) * h, 1), round(bw * w, 1), round(bh * h, 1)]
                a["attrs"]["size"] = "tiny" if bh * h < 16 else "small" if bh * h < 48 else "large"
    (pt / "labels.json").write_text(json.dumps(meta, indent=1))
    (args.root / "rotation_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: len(v) for k, v in report.items()}))


if __name__ == "__main__":
    main()
