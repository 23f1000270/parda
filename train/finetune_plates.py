"""Fine-tune a plate detector on Indian number plates and export it for Parda.

    pip install ultralytics
    python train/prepare_roboflow.py --src downloads/ --out data/plates
    python train/finetune_plates.py --data data/plates/data.yaml

A GPU makes this take minutes. It also runs on a plain CPU, slowly: our first
model trained overnight on two cores. If a CPU run is stopped, continue it with
--resume.

The dataset file is a standard Ultralytics yaml with one class, "plate". Mix in
as much variety as you can find: two-line plates, two-wheeler and auto-rickshaw
plates, yellow commercial, green EV, BH-series and hand-painted plates, night
and rain. Check every dataset's licence before training on it.

Licence note: Ultralytics and the weights it produces are AGPL-3.0. Parda's own
code stays Apache-2.0; a model trained here is a separate file under AGPL-3.0.

The exported model (ONNX with built-in NMS) plugs straight into Parda:

    parda video in.mp4 -o out.mp4 --plate-model runs/plates/indian/weights/best.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Ultralytics dataset yaml")
    ap.add_argument("--base", default="yolo11n.pt", help="starting weights")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=416, help="training size")
    ap.add_argument("--export-size", type=int, default=384, help="ONNX input size (Parda's default plate pass)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--device", default=None, help="e.g. cpu or 0; default picks a GPU if there is one")
    ap.add_argument("--cache", default="ram", help="ram, disk or none")
    ap.add_argument("--name", default="indian")
    ap.add_argument("--resume", action="store_true", help="continue an interrupted run")
    args = ap.parse_args()

    from ultralytics import YOLO  # imported here so the package does not depend on it

    # An absolute project path stops Ultralytics from nesting it under runs/detect/.
    project = Path("runs/plates").resolve()
    run_dir = project / args.name
    if args.resume:
        YOLO(str(run_dir / "weights" / "last.pt")).train(resume=True)
    else:
        YOLO(args.base).train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            device=args.device,
            cache=False if args.cache == "none" else args.cache,
            project=str(project),
            name=args.name,
            exist_ok=True,
            # Plates are small and often at the frame edge; keep mosaic, add scale jitter.
            scale=0.6,
            mosaic=1.0,
            close_mosaic=10,
            # Night and glare: widen brightness jitter.
            hsv_v=0.55,
            patience=20,
            plots=False,
        )
    best = YOLO(str(run_dir / "weights" / "best.pt"))
    path = best.export(format="onnx", imgsz=args.export_size, nms=True, simplify=True, opset=17)
    print("exported", path)


if __name__ == "__main__":
    main()
