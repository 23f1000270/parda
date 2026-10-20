"""Command line interface: `parda image|video|live|bench|convert-cvat|fetch-models`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__, models
from .pipeline import Anonymizer, Config

VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".ts"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _config(args) -> Config:
    return Config(
        face_model=args.face_model,
        plate_model=args.plate_model,
        face_thresh=args.face_thresh,
        plate_thresh=args.plate_thresh,
        mode=args.mode,
        faces=not args.no_faces,
        plates=not args.no_plates,
        plate_tiles=args.thorough,
    )


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--face-model", default=models.DEFAULT_FACE)
    p.add_argument("--plate-model", default=models.DEFAULT_PLATE)
    p.add_argument("--face-thresh", type=float, default=0.4)
    p.add_argument("--plate-thresh", type=float, default=None,
                   help="default: the threshold registered for the plate model (0.2 for others)")
    p.add_argument("--mode", choices=["blocks", "solid", "black"], default="blocks")
    p.add_argument("--no-faces", action="store_true")
    p.add_argument("--no-plates", action="store_true")
    p.add_argument("--thorough", action="store_true",
                   help="also scan plates in native-resolution tiles (slower, finds distant plates)")


def _progress(n: int, total: int | None) -> None:
    if n % 25 == 0 or (total and n == total):
        msg = f"\r  frame {n}" + (f"/{total}" if total else "")
        print(msg, end="", file=sys.stderr, flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="parda", description="Blur faces and number plates on-device.")
    ap.add_argument("--version", action="version", version=f"parda {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("image", help="anonymise images (files or a folder)")
    p.add_argument("inputs", nargs="+", type=Path)
    p.add_argument("-o", "--out", type=Path, required=True, help="output folder")
    _common(p)

    p = sub.add_parser("video", help="anonymise a video file")
    p.add_argument("input", type=Path)
    p.add_argument("-o", "--out", type=Path, required=True)
    _common(p)

    p = sub.add_parser("live", help="anonymise a camera stream before anything is saved")
    p.add_argument("--source", default="0", help="camera index, stream URL or 'picamera2'")
    p.add_argument("--out-dir", type=Path, default=Path("parda_recordings"))
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=float, default=10.0)
    p.add_argument("--segment", type=int, default=60, help="seconds per file")
    p.add_argument("--seconds", type=float, default=None, help="stop after this long")
    p.add_argument("--preview", action="store_true")
    _common(p)

    p = sub.add_parser("bench", help="score Parda (or another tool) on a labelled set")
    p.add_argument("dataset", type=Path)
    p.add_argument("--predictions", type=Path, help="score this JSON of masked boxes instead of running Parda")
    p.add_argument("--save-predictions", type=Path)
    p.add_argument("--json", type=Path, help="write the results as JSON too")
    _common(p)

    p = sub.add_parser("convert-cvat", help="convert a CVAT COCO 1.0 export to labels.json")
    p.add_argument("src", type=Path)
    p.add_argument("dst", type=Path)

    sub.add_parser("fetch-models", help="download and verify the default models")

    args = ap.parse_args(argv)

    if args.cmd == "fetch-models":
        for name in models.MODELS:
            print(name, "->", models.resolve(name))
        return 0

    if args.cmd == "convert-cvat":
        from .bench import convert_cvat_coco
        convert_cvat_coco(args.src, args.dst)
        print("wrote", args.dst)
        return 0

    if args.cmd == "bench":
        from .bench import load_dataset, run_parda, score
        from .metrics import to_markdown
        images = load_dataset(args.dataset)
        fps = None
        if args.predictions:
            preds = json.loads(args.predictions.read_text())
        else:
            preds, fps = run_parda(Anonymizer(_config(args)), args.dataset, images)
            if args.save_predictions:
                args.save_predictions.write_text(json.dumps(preds))
        res = score(images, preds)
        print(to_markdown(res))
        if fps:
            print(f"Speed: {fps:.2f} images/s on this machine")
        if args.json:
            args.json.write_text(json.dumps({
                "frames": res.frames, "over_mask": res.over_mask, "images_per_s": fps,
                "groups": {k: {"objects": t.total, **{f"protected@{l}": t.recall(l) for l in res.levels}}
                           for k, t in res.by_group.items()},
            }, indent=1))
        return 0

    anon = Anonymizer(_config(args))

    if args.cmd == "image":
        from .video import anonymize_image
        files = []
        for p in args.inputs:
            files += sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXT) if p.is_dir() else [p]
        args.out.mkdir(parents=True, exist_ok=True)
        for f in files:
            n = anonymize_image(anon, f, args.out / f.name)
            print(f"{f.name}: masked {n} regions")
        return 0

    if args.cmd == "video":
        from .video import anonymize_video
        rep = anonymize_video(anon, args.input, args.out, progress=_progress)
        print(f"\n{rep.frames} frames in {rep.seconds:.1f}s ({rep.fps:.1f} fps), "
              f"fail-closed frames: {rep.failsafe_frames}. Saved {args.out} (no audio, no metadata).")
        return 0

    if args.cmd == "live":
        from .live import run_live
        run_live(anon, args.out_dir, args.source, args.width, args.height, args.fps,
                 args.segment, args.seconds, args.preview)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
