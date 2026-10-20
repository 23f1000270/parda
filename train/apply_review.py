"""Apply a manual review file to the prepared test split (see train/review/).

    python train/apply_review.py data/plates train/review/test_review.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("review", type=Path)
    args = ap.parse_args()
    rev = json.loads(args.review.read_text())
    pt = args.root / "parda_test"
    meta = json.loads((pt / "labels.json").read_text())
    by_file = {Path(im["file"]).name: im for im in meta["images"]}
    for name in rev.get("rotate_cw90", []):
        src = args.root / "test" / "images" / name
        img = cv2.rotate(cv2.imread(str(src)), cv2.ROTATE_90_CLOCKWISE)
        cv2.imwrite(str(src), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(pt / "frames" / name), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        h, w = img.shape[:2]
        im = by_file[name]
        im["width"], im["height"] = w, h
        rel = [tuple(map(float, l.split()[1:5]))
               for l in (args.root / "test" / "labels" / (Path(name).stem + ".txt")).read_text().split("\n") if l.strip()]
        anns = [a for a in meta["annotations"] if a["image_id"] == im["id"]]
        for a, (cx, cy, bw, bh) in zip(anns, rel):
            a["bbox"] = [round((cx - bw / 2) * w, 1), round((cy - bh / 2) * h, 1), round(bw * w, 1), round(bh * h, 1)]
            a["attrs"]["size"] = "tiny" if bh * h < 16 else "small" if bh * h < 48 else "large"
    drop_ids = {by_file[n]["id"] for n in rev.get("remove", [])}
    for name in rev.get("remove", []):
        for p in (args.root / "test" / "images" / name, args.root / "test" / "labels" / (Path(name).stem + ".txt"),
                  pt / "frames" / name):
            p.unlink(missing_ok=True)
    meta["images"] = [im for im in meta["images"] if im["id"] not in drop_ids]
    meta["annotations"] = [a for a in meta["annotations"] if a["image_id"] not in drop_ids]
    (pt / "labels.json").write_text(json.dumps(meta, indent=1))
    print(f"rotated {len(rev.get('rotate_cw90', []))}, removed {len(drop_ids)}; "
          f"test now {len(meta['images'])} images, {len(meta['annotations'])} plates")


if __name__ == "__main__":
    main()
