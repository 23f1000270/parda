"""Video and image file processing.

Output videos never carry over the input's audio track or metadata. Dashcam
files often embed GPS tracks and device serials in metadata, and voices on the
audio track are personal data too.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import cv2

from .pipeline import Anonymizer


@dataclass
class VideoReport:
    frames: int
    seconds: float
    failsafe_frames: int
    mean_latency_ms: float

    @property
    def fps(self) -> float:
        return self.frames / self.seconds if self.seconds else 0.0


def anonymize_image(anon: Anonymizer, src: Path, dst: Path) -> int:
    img = cv2.imread(str(src))
    if img is None:
        raise ValueError(f"Could not read image {src}")
    anon.reset()
    res = anon.process(img, use_memory=False)
    # cv2.imwrite does not copy EXIF, so GPS tags in the source are dropped.
    if not cv2.imwrite(str(dst), res.image):
        raise ValueError(f"Could not write {dst}")
    return len(res.masked_boxes)


def anonymize_video(anon: Anonymizer, src: Path, dst: Path, progress=None) -> VideoReport:
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise ValueError(f"Could not open video {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None

    anon.reset()
    tmpdir = Path(tempfile.mkdtemp(prefix="parda_"))
    raw_out = tmpdir / "masked.mp4"
    writer = cv2.VideoWriter(str(raw_out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    n = failsafe = 0
    lat = 0.0
    t0 = time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            res = anon.process(frame)
            writer.write(res.image)  # only masked frames are ever written
            n += 1
            lat += res.latency_ms
            failsafe += res.failsafe is not None
            if progress:
                progress(n, total)
    finally:
        cap.release()
        writer.release()
    elapsed = time.perf_counter() - t0
    _finalise(raw_out, dst)
    shutil.rmtree(tmpdir, ignore_errors=True)
    return VideoReport(n, elapsed, failsafe, lat / max(n, 1))


def _finalise(raw: Path, dst: Path) -> None:
    """Re-encode to H.264 without audio or metadata when ffmpeg is available."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        shutil.move(str(raw), dst)
        return
    cmd = [
        ffmpeg, "-y", "-loglevel", "error", "-i", str(raw),
        "-map", "0:v:0", "-an", "-map_metadata", "-1", "-map_chapters", "-1",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart",
        str(dst),
    ]
    subprocess.run(cmd, check=True)
