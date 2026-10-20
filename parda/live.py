"""Live capture: anonymise frames before they ever reach storage.

Frames come off the camera into memory, get masked, and only then are written
to disk. If processing falls behind, frames are dropped, never written raw.
On a Raspberry Pi 5 with the Camera Module, Picamera2 is used when installed;
otherwise any OpenCV camera index works (USB webcams, laptops).
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import cv2

from .pipeline import Anonymizer


def _open_camera(source: str, width: int, height: int):
    if source == "picamera2":
        from picamera2 import Picamera2  # type: ignore

        cam = Picamera2()
        cam.configure(cam.create_video_configuration(main={"size": (width, height), "format": "RGB888"}))
        cam.start()
        return lambda: (True, cam.capture_array()), cam.stop
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap.read, cap.release


def run_live(
    anon: Anonymizer,
    out_dir: Path,
    source: str = "0",
    width: int = 1280,
    height: int = 720,
    fps: float = 10.0,
    segment_seconds: int = 60,
    max_seconds: float | None = None,
    preview: bool = False,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    read, close = _open_camera(source, width, height)
    writer = None
    seg_start = 0.0
    start = time.monotonic()
    frames = dropped = 0
    try:
        while max_seconds is None or time.monotonic() - start < max_seconds:
            ok, frame = read()
            if not ok:
                break
            t = time.monotonic()
            if writer is None or t - seg_start >= segment_seconds:
                if writer is not None:
                    writer.release()
                name = out_dir / f"parda_{datetime.now():%Y%m%d_%H%M%S}.mp4"
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(str(name), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                seg_start = t
            res = anon.process(frame)
            del frame  # the raw frame is not kept past this point
            if res.latency_ms > 1000 / fps * 3:
                dropped += 1  # too slow: skip writing rather than fall behind
                continue
            writer.write(res.image)
            frames += 1
            if preview:
                cv2.imshow("parda (masked)", res.image)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        if writer is not None:
            writer.release()
        close()
        if preview:
            cv2.destroyAllWindows()
    print(f"wrote {frames} masked frames, dropped {dropped} slow frames, to {out_dir}")
