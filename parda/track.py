"""Mask memory: keep masking an object for a few frames after the detector loses it.

Detectors flicker. A face found in frame 10 and 12 but missed in frame 11 would
be exposed for one frame, which is enough for anyone scrubbing through the
video. MaskMemory holds every track for `hold_frames` frames after its last
detection, moves it along its recent velocity and grows it a little each frame
it goes unseen, so uncertainty is always covered by a bigger mask, never a
smaller one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .detect import Detection


def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


@dataclass
class _Track:
    box: np.ndarray
    label: str
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))
    missed: int = 0
    hits: int = 1


class MaskMemory:
    def __init__(self, hold_frames: int = 8, match_iou: float = 0.2, grow_per_frame: float = 0.06):
        self.hold_frames = hold_frames
        self.match_iou = match_iou
        self.grow = grow_per_frame
        self.tracks: list[_Track] = []

    def reset(self) -> None:
        self.tracks.clear()

    def update(self, detections: list[Detection]) -> list[tuple[np.ndarray, str]]:
        """Feed this frame's detections, get back every box that must be masked."""
        # Predict where existing tracks are now.
        for t in self.tracks:
            t.box = t.box + np.array([*t.velocity, *t.velocity])

        unmatched = list(range(len(detections)))
        # Greedy matching, best IoU first.
        pairs = []
        for ti, t in enumerate(self.tracks):
            for di in unmatched:
                d = detections[di]
                if d.label == t.label:
                    v = iou(t.box, d.box)
                    if v >= self.match_iou:
                        pairs.append((v, ti, di))
        pairs.sort(reverse=True)
        used_t, used_d = set(), set()
        for _, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            t = self.tracks[ti]
            new = np.array(detections[di].box, dtype=float)
            c_old = (t.box[:2] + t.box[2:]) / 2
            c_new = (new[:2] + new[2:]) / 2
            t.velocity = 0.5 * t.velocity + 0.5 * (c_new - c_old)
            t.box = new
            t.missed = 0
            t.hits += 1

        for ti, t in enumerate(self.tracks):
            if ti not in used_t:
                t.missed += 1
                w, h = t.box[2] - t.box[0], t.box[3] - t.box[1]
                g = self.grow / 2
                t.box = t.box + np.array([-w * g, -h * g, w * g, h * g])

        for di, d in enumerate(detections):
            if di not in used_d:
                self.tracks.append(_Track(box=np.array(d.box, dtype=float), label=d.label))

        self.tracks = [t for t in self.tracks if t.missed <= self.hold_frames]
        return [(t.box.copy(), t.label) for t in self.tracks]
