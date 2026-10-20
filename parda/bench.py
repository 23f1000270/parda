"""Benchmark runner for the Indian roads test set (or any set in the same format).

Dataset layout::

    my_set/
      labels.json
      frames/0001.jpg ...

labels.json follows a small COCO-like schema::

    {
      "images": [{"id": 1, "file": "frames/0001.jpg", "width": 1920, "height": 1080,
                  "attrs": {"light": "night"}}],
      "annotations": [{"image_id": 1, "label": "plate", "bbox": [x, y, w, h],
                       "attrs": {"plate_type": "two_line", "vehicle": "two_wheeler"}}]
    }

CVAT "COCO 1.0" exports can be converted with ``parda convert-cvat``.

Other tools are scored the same way: export their masked regions as
``{"frames/0001.jpg": [[x1, y1, x2, y2], ...]}`` and pass ``--predictions``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2

from .metrics import EvalResult, evaluate
from .pipeline import Anonymizer


def load_dataset(root: Path):
    meta = json.loads((root / "labels.json").read_text())
    by_img = {im["id"]: {**im, "gts": []} for im in meta["images"]}
    for a in meta["annotations"]:
        x, y, w, h = a["bbox"]
        attrs = dict(a.get("attrs") or {})
        attrs.update({f"img_{k}": v for k, v in (by_img[a["image_id"]].get("attrs") or {}).items()})
        by_img[a["image_id"]]["gts"].append((a["label"], (x, y, x + w, y + h), attrs))
    return list(by_img.values())


def run_parda(anon: Anonymizer, root: Path, images) -> tuple[dict, float]:
    preds = {}
    t0 = time.perf_counter()
    for im in images:
        img = cv2.imread(str(root / im["file"]))
        if img is None:
            raise FileNotFoundError(root / im["file"])
        anon.reset()
        res = anon.process(img, use_memory=False)
        preds[im["file"]] = [list(map(int, b)) for b in res.masked_boxes]
    secs = time.perf_counter() - t0
    return preds, len(images) / secs if secs else 0.0


def score(images, preds: dict, levels=(0.8, 0.95)) -> EvalResult:
    samples = ((im["width"], im["height"], im["gts"], preds.get(im["file"], [])) for im in images)
    return evaluate(samples, levels)


def convert_cvat_coco(src: Path, dst: Path, frames_prefix: str = "frames/") -> None:
    """Convert a CVAT COCO 1.0 export (categories 'face' and 'plate', attributes
    stored per annotation) into Parda's labels.json."""
    coco = json.loads(src.read_text())
    cats = {c["id"]: c["name"].lower() for c in coco["categories"]}
    out = {
        "images": [
            {"id": im["id"], "file": frames_prefix + Path(im["file_name"]).name,
             "width": im["width"], "height": im["height"], "attrs": im.get("attributes", {})}
            for im in coco["images"]
        ],
        "annotations": [
            {"image_id": a["image_id"], "label": cats[a["category_id"]], "bbox": a["bbox"],
             "attrs": {k: v for k, v in (a.get("attributes") or {}).items() if k != "occluded"}}
            for a in coco["annotations"] if cats[a["category_id"]] in ("face", "plate")
        ],
    }
    dst.write_text(json.dumps(out, indent=1))
