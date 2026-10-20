"""Fine-tune a plate detector on Indian number plates and export it for Parda.

Run this on a machine with a GPU (an RTX 4060 laptop is plenty):

    pip install ultralytics
    python train/finetune_plates.py --data path/to/plates.yaml --epochs 120

plates.yaml is a standard Ultralytics dataset file with one class, "plate".
Mix general plate data with Indian-specific images so the model learns:
two-line plates, two-wheeler and auto-rickshaw plates, yellow commercial,
green EV, BH-series and hand-painted plates, at night and in rain.
Check every dataset's licence before training on it.

The exported model (ONNX with built-in NMS) plugs straight into Parda:

    parda video in.mp4 -o out.mp4 --plate-model runs/plates/indian/weights/best.onnx
"""

from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Ultralytics dataset yaml")
    ap.add_argument("--base", default="yolo11n.pt", help="starting weights")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--name", default="indian")
    args = ap.parse_args()

    from ultralytics import YOLO  # imported here so the package does not depend on it

    model = YOLO(args.base)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project="runs/plates",
        name=args.name,
        # Plates are small and often at the frame edge; keep mosaic, add scale jitter.
        scale=0.6,
        mosaic=1.0,
        close_mosaic=15,
        # Night and glare: widen brightness jitter.
        hsv_v=0.55,
        patience=30,
    )
    best = YOLO(f"runs/plates/{args.name}/weights/best.pt")
    path = best.export(format="onnx", imgsz=args.imgsz, nms=True, simplify=True, opset=17)
    print("exported", path)


if __name__ == "__main__":
    main()
