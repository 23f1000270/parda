"""The anonymisation pipeline: detect, remember, mask. Fails closed."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from . import models
from .detect import Detection, FaceDetector, PlateDetector
from .mask import apply_masks, expand, full_frame
from .track import MaskMemory


@dataclass
class Config:
    face_model: str = models.DEFAULT_FACE
    plate_model: str = models.DEFAULT_PLATE
    face_thresh: float = 0.4
    plate_thresh: float | None = None  # None: use the threshold registered for the plate model
    plate_tile_thresh: float | None = None  # None: registered value, else same as plate_thresh
    face_tiles: bool = True        # native-scale tiles catch small faces in crowds
    plate_tiles: bool = False      # slower; turn on for distant plates in batch jobs
    face_margin: float = 0.18      # faces: hair and ears identify people too
    plate_margin: float = 0.12
    mode: str = "blocks"           # blocks | solid | black
    hold_frames: int = 8           # keep masking this many frames after a track is lost
    faces: bool = True
    plates: bool = True
    fail_closed: bool = True       # on any detector error, hide the whole frame


@dataclass
class FrameResult:
    image: np.ndarray
    detections: list[Detection] = field(default_factory=list)
    masked_boxes: list[tuple[int, int, int, int]] = field(default_factory=list)
    latency_ms: float = 0.0
    failsafe: str | None = None


class Anonymizer:
    def __init__(self, config: Config | None = None, seed: int | None = None,
                 face_detector=None, plate_detector=None):
        """Detectors are built from the config unless passed in (handy for tests
        or for plugging in a fine-tuned model object)."""
        self.cfg = config or Config()
        if face_detector is not None:
            self.face = face_detector
        elif self.cfg.faces:
            self.face = FaceDetector(str(models.resolve(self.cfg.face_model)), self.cfg.face_thresh,
                                     tiled=self.cfg.face_tiles)
        else:
            self.face = None
        if plate_detector is not None:
            self.plate = plate_detector
        elif self.cfg.plates:
            thresh = self.cfg.plate_thresh
            if thresh is None:
                thresh = models.plate_thresh_for(self.cfg.plate_model)
            tile_thresh = self.cfg.plate_tile_thresh
            if tile_thresh is None and self.cfg.plate_thresh is None:
                tile_thresh = models.plate_tile_thresh_for(self.cfg.plate_model)
            self.plate = PlateDetector(str(models.resolve(self.cfg.plate_model)), thresh, tile_thresh=tile_thresh,
                                       tiled=self.cfg.plate_tiles)
        else:
            self.plate = None
        self.memory = MaskMemory(hold_frames=self.cfg.hold_frames)
        self.rng = np.random.default_rng(seed)

    def reset(self) -> None:
        """Call between unrelated images or videos so masks do not carry over."""
        self.memory.reset()

    def detect(self, frame: np.ndarray) -> list[Detection]:
        dets: list[Detection] = []
        if self.face is not None:
            dets += self.face.detect(frame)
        if self.plate is not None:
            dets += self.plate.detect(frame)
        return dets

    def process(self, frame: np.ndarray, use_memory: bool = True) -> FrameResult:
        t0 = time.perf_counter()
        h, w = frame.shape[:2]
        try:
            dets = self.detect(frame)
        except Exception as exc:  # noqa: BLE001 - any failure must not leak the frame
            if not self.cfg.fail_closed:
                raise
            return FrameResult(full_frame(frame), failsafe=f"detector_error: {exc}",
                               latency_ms=(time.perf_counter() - t0) * 1e3)

        tracked = self.memory.update(dets) if use_memory else [(np.array(d.box), d.label) for d in dets]
        boxes = []
        for box, label in tracked:
            margin = self.cfg.face_margin if label == "face" else self.cfg.plate_margin
            boxes.append(expand(box, margin, w, h))
        out = apply_masks(frame, boxes, mode=self.cfg.mode, rng=self.rng)
        return FrameResult(out, dets, boxes, (time.perf_counter() - t0) * 1e3)
