"""Merge several Roboflow Universe plate datasets into one clean training set.

Public Indian plate datasets on Roboflow Universe overlap heavily: the same
photos and video frames are re-uploaded and forked under different names,
often with augmented copies. Training and testing on such a mix without care
puts near-copies of test images into training, and the test score then means
nothing. This script:

1. reads each export (YOLO format: data.yaml plus train/valid/test folders),
2. keeps only plate boxes, whatever each dataset called its plate class,
3. groups images that must stay together: the same original file (Roboflow
   keeps the original name before ".rf."), frames of the same video, photos
   of the same plate number, and near-identical pictures by perceptual hash,
   mirrored copies included,
4. keeps one copy of each exact or near duplicate,
5. splits whole groups into train, val and test, so no group is split.

It writes a YOLO dataset for training and a copy of the test split in
Parda's benchmark format (labels.json plus frames/), so the same test images
score both the stock model and the fine-tuned one.

    python train/prepare_roboflow.py --src downloads/ --out data/plates
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageOps

try:
    import imagehash
except ImportError as e:  # pragma: no cover
    raise SystemExit("pip install imagehash") from e

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# Classes that are not plates in the mixed datasets. Everything else is a plate:
# some datasets name each class after the plate number it shows.
NOT_PLATE = re.compile(r"helmet|vehicle|^car$|cars|person|rider|bike|truck|motor|face|noise|logo|city|year", re.I)
NEAR_DUP_BITS = 6  # pHash Hamming distance treated as the same picture
MAX_SIDE = 1920    # phone photos are shrunk to dashcam-like frame sizes


def original_stem(name: str) -> str:
    stem = Path(name).stem
    stem = re.split(r"\.rf\.", stem)[0]
    stem = re.sub(r"_(jpe?g|png|bmp|webp)$", "", stem, flags=re.I)
    return stem.lower()


def group_key(stem: str) -> str:
    """Coarse group: frames of one video, or photos of one plate number."""
    m = re.match(r"(video\d+)_\d+", stem)
    if m:
        return m.group(1)
    m = re.search(r"([a-z]{2}\d{1,2}[a-z]{0,3}\d{3,4})", stem.replace("-", "").replace(" ", ""))
    if m:
        return "plate:" + m.group(1)
    return "file:" + stem


def read_labels(path: Path, keep: set[int]) -> list[tuple[float, float, float, float]]:
    boxes = []
    if not path.exists():
        return boxes
    for line in path.read_text().split("\n"):
        p = line.split()
        if len(p) < 5 or int(float(p[0])) not in keep:
            continue
        v = list(map(float, p[1:]))
        if len(v) == 4:
            cx, cy, w, h = v
        else:  # polygon: x1 y1 x2 y2 ...
            xs, ys = v[0::2], v[1::2]
            cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
            w, h = max(xs) - min(xs), max(ys) - min(ys)
        if w > 0.002 and h > 0.002:
            boxes.append((cx, cy, w, h))
    return boxes


def scan(src: Path):
    items = []
    stats = {}
    for yml in sorted(src.glob("*/data.yaml")):
        root = yml.parent
        names = yaml.safe_load(yml.read_text()).get("names", [])
        if isinstance(names, dict):
            names = [names[k] for k in sorted(names)]
        keep = {i for i, n in enumerate(names) if not NOT_PLATE.search(str(n))}
        readme = root / "README.roboflow.txt"
        # Some exports squash every picture to a square. Those are fine to train
        # on but do not look like real camera frames, so they never go to test.
        stretched = readme.exists() and "Stretch" in readme.read_text()
        n_img = n_box = 0
        for img in sorted(root.rglob("*")):
            if img.suffix.lower() not in IMG_EXT or img.parent.name != "images":
                continue
            lab = img.parent.parent / "labels" / (img.stem + ".txt")
            boxes = read_labels(lab, keep)
            items.append({"src": root.name, "path": img, "boxes": boxes, "stem": original_stem(img.name),
                          "stretched": stretched})
            n_img += 1
            n_box += len(boxes)
        stats[root.name] = {"images": n_img, "plates": n_box, "classes_kept": len(keep), "classes": len(names),
                            "stretched": stretched}
    return items, stats


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, i):
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]
            i = self.p[i]
        return i

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.p[b] = a


def load(path: Path) -> Image.Image:
    # Phone photos often store rotation as an EXIF tag. The labels follow the
    # rotated view, so the pixels must be rotated before anything else.
    return ImageOps.exif_transpose(Image.open(path))


def phash_pair(path: Path):
    im = load(path).convert("L")
    return int(str(imagehash.phash(im)), 16), int(str(imagehash.phash(ImageOps.mirror(im))), 16)


def popcount64(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.uint64)
    c = np.zeros(x.shape, dtype=np.uint8)
    for s in range(0, 64, 8):
        c += _POP8[((x >> np.uint64(s)) & np.uint64(0xFF)).astype(np.uint8)]
    return c


_POP8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True, help="folder of extracted Roboflow YOLO exports")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--val", type=float, default=0.10)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    items, stats = scan(args.src)
    print(f"read {len(items)} images from {len(stats)} datasets")

    # Exact duplicates by content hash.
    seen, uniq = {}, []
    for it in items:
        h = hashlib.sha1(it["path"].read_bytes()).hexdigest()
        if h in seen:
            continue
        seen[h] = True
        uniq.append(it)
    print(f"{len(uniq)} after removing byte-identical copies")

    hashes = np.array([phash_pair(it["path"]) for it in uniq], dtype=np.uint64)
    n = len(uniq)
    uf = UnionFind(n)
    by_key = defaultdict(list)
    for i, it in enumerate(uniq):
        by_key[it["stem"]].append(i)
        by_key[group_key(it["stem"])].append(i)
    for idx in by_key.values():
        for j in idx[1:]:
            uf.union(idx[0], j)
    # Near-identical pictures, mirrored copies included.
    a = hashes[:, 0]
    for i in range(n):
        d = np.minimum(popcount64(a[i + 1:] ^ a[i]), popcount64(hashes[i + 1:, 1] ^ a[i]))
        for j in np.nonzero(d <= NEAR_DUP_BITS)[0]:
            uf.union(i, i + 1 + int(j))

    # Within each group, drop near-duplicates (keep the copy with the most plates).
    groups = defaultdict(list)
    for i in range(n):
        groups[uf.find(i)].append(i)
    kept = []
    for members in groups.values():
        # Prefer the unsquashed copy, then the copy with the most plates labelled.
        members.sort(key=lambda i: (uniq[i]["stretched"], -len(uniq[i]["boxes"])))
        chosen = []
        for i in members:
            dup = any(min(bin(int(hashes[i, 0] ^ hashes[k, 0])).count("1"),
                          bin(int(hashes[i, 1] ^ hashes[k, 0])).count("1")) <= NEAR_DUP_BITS for k in chosen)
            if not dup:
                chosen.append(i)
        kept.append([i for i in chosen if uniq[i]["boxes"]])
    kept = [g for g in kept if g]
    n_kept = sum(len(g) for g in kept)
    print(f"{n_kept} distinct labelled images in {len(kept)} groups")

    rng = random.Random(args.seed)
    rng.shuffle(kept)
    target_test, target_val = args.test * n_kept, args.val * n_kept
    split = {"test": [], "val": [], "train": []}
    filled = {"test": 0, "val": 0}
    slack = 0.02 * n_kept  # a big video group must not swallow the whole test split
    for g in kept:
        if any(uniq[i]["stretched"] for i in g):
            split["train"].append(g)
            continue
        for name, target in (("test", target_test), ("val", target_val)):
            if filled[name] + len(g) <= target + slack:
                split[name].append(g)
                filled[name] += len(g)
                break
        else:
            split["train"].append(g)

    out = args.out
    if out.exists():
        shutil.rmtree(out)
    parda_test = {"images": [], "annotations": []}
    counts = {}
    for name, gs in split.items():
        (out / name / "images").mkdir(parents=True)
        (out / name / "labels").mkdir(parents=True)
        k = 0
        for g in gs:
            for i in g:
                it = uniq[i]
                k += 1
                fname = f"{name}_{k:05d}.jpg"
                im = load(it["path"]).convert("RGB")
                if max(im.size) > MAX_SIDE:  # labels are relative, so they survive the resize
                    im.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
                im.save(out / name / "images" / fname, quality=95)
                (out / name / "labels" / (fname[:-4] + ".txt")).write_text(
                    "\n".join(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for cx, cy, w, h in it["boxes"]) + "\n")
                if name == "test":
                    W, H = im.size
                    img_id = len(parda_test["images"]) + 1
                    parda_test["images"].append({"id": img_id, "file": f"frames/{fname}", "width": W, "height": H,
                                                 "attrs": {"source": it["src"]}})
                    for cx, cy, w, h in it["boxes"]:
                        bw, bh = w * W, h * H
                        size = "tiny" if bh < 16 else "small" if bh < 48 else "large"
                        parda_test["annotations"].append({
                            "image_id": img_id, "label": "plate",
                            "bbox": [round((cx - w / 2) * W, 1), round((cy - h / 2) * H, 1), round(bw, 1), round(bh, 1)],
                            "attrs": {"size": size}})
        counts[name] = {"images": k, "groups": len(gs),
                        "plates": sum(len(uniq[i]["boxes"]) for g in gs for i in g)}
    (out / "data.yaml").write_text(yaml.safe_dump({
        "path": str(out.resolve()), "train": "train/images", "val": "val/images", "test": "test/images",
        "names": {0: "plate"}}))
    pt = out / "parda_test"
    (pt / "frames").mkdir(parents=True)
    for im in parda_test["images"]:
        shutil.copy(out / "test" / "images" / Path(im["file"]).name, pt / im["file"])
    (pt / "labels.json").write_text(json.dumps(parda_test, indent=1))
    report = {"sources": stats, "read": len(items), "byte_unique": len(uniq), "kept": n_kept, "split": counts}
    (out / "prepare_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report["split"], indent=1))


if __name__ == "__main__":
    main()
