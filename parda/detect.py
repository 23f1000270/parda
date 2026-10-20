"""Face and number-plate detectors running on ONNX Runtime.

Both detectors take a BGR uint8 image of any size and return boxes in the
original image coordinates. Thresholds default low on purpose: for
anonymisation a missed face is a privacy leak, while an extra box only costs a
little picture quality.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import onnxruntime as ort


@dataclass(frozen=True)
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    label: str  # "face" or "plate"

    @property
    def box(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)


def _session(path: str) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                 if p in ort.get_available_providers()]
    return ort.InferenceSession(path, sess_options=opts, providers=providers)


def letterbox(img: np.ndarray, size: int, pad_value: int = 114):
    """Resize keeping aspect ratio and pad to size x size.

    Returns the padded image, the scale factor and the (left, top) padding.
    """
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR) if (nw, nh) != (w, h) else img
    left = (size - nw) // 2
    top = (size - nh) // 2
    out = np.full((size, size, 3), pad_value, dtype=np.uint8)
    out[top:top + nh, left:left + nw] = resized
    return out, r, (left, top)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> np.ndarray:
    """Plain greedy non-maximum suppression. boxes are x1, y1, x2, y2."""
    if len(boxes) == 0:
        return np.zeros((0,), dtype=np.int64)
    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thresh]
    return np.array(keep, dtype=np.int64)


def decode_yunet(outputs: dict[str, np.ndarray], input_size: int, score_thresh: float):
    """Decode YuNet (2023mar) multi-stride heads into boxes and scores.

    For each stride s the head predicts, per grid cell (row, col), a class score,
    an objectness score and a box (dx, dy, log w, log h) in units of s.
    """
    boxes, scores = [], []
    for s in (8, 16, 32):
        cols = rows = input_size // s
        cls = outputs[f"cls_{s}"].reshape(-1)
        obj = outputs[f"obj_{s}"].reshape(-1)
        bb = outputs[f"bbox_{s}"].reshape(-1, 4)
        score = np.sqrt(np.clip(cls, 0, 1) * np.clip(obj, 0, 1))
        idx = np.nonzero(score >= score_thresh)[0]
        if idx.size == 0:
            continue
        r, c = np.divmod(idx, cols)
        cx = (c + bb[idx, 0]) * s
        cy = (r + bb[idx, 1]) * s
        w = np.exp(bb[idx, 2]) * s
        h = np.exp(bb[idx, 3]) * s
        boxes.append(np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1))
        scores.append(score[idx])
        assert rows == cols
    if not boxes:
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32)
    return np.concatenate(boxes).astype(np.float32), np.concatenate(scores).astype(np.float32)


def tile_origins(h: int, w: int, size: int, overlap: float = 0.15) -> list[tuple[int, int]]:
    """Top-left corners of size x size tiles that cover an h x w frame with overlap."""
    step = max(1, int(size * (1 - overlap)))

    def axis(n: int) -> list[int]:
        if n <= size:
            return [0]
        pos = list(range(0, n - size + 1, step))
        if pos[-1] != n - size:
            pos.append(n - size)
        return pos

    return [(x, y) for y in axis(h) for x in axis(w)]


class _TiledDetector:
    """Shared logic: one letterboxed pass for large objects, plus native-scale
    tiles so small faces and distant plates are not shrunk out of existence."""

    label = ""
    pad_value = 0

    def __init__(self, model_path: str, score_thresh: float, nms_thresh: float, tiled: bool,
                 tile_thresh: float | None = None):
        self.sess = _session(model_path)
        inp = self.sess.get_inputs()[0]
        self.input_name = inp.name
        self.size = int(inp.shape[2])
        self.out_names = [o.name for o in self.sess.get_outputs()]
        self.score_thresh = score_thresh
        # Tiles look at many more small regions, so a model run at a low threshold
        # can be given a stricter one there to keep false alarms down.
        self.tile_thresh = score_thresh if tile_thresh is None else max(score_thresh, tile_thresh)
        self.nms_thresh = nms_thresh
        self.tiled = tiled

    def _infer(self, square: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def _pass(self, img: np.ndarray, r: float, pad: tuple[int, int], offset=(0, 0)):
        boxes, scores = self._infer(img)
        if len(boxes):
            boxes = boxes.copy()
            boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad[0]) / r + offset[0]
            boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad[1]) / r + offset[1]
        return boxes, scores

    def detect(self, img_bgr: np.ndarray) -> list[Detection]:
        h, w = img_bgr.shape[:2]
        all_b, all_s = [], []
        # Pass 1: whole frame letterboxed to the model size.
        padded, r, pad = letterbox(img_bgr, self.size, self.pad_value)
        b, s = self._pass(padded, r, pad)
        all_b.append(b)
        all_s.append(s)
        # Pass 2 (optional): native-scale tiles. Models score small objects
        # differently at different scales, so we take the union of both views.
        if self.tiled:
            if max(h, w) <= self.size:
                square = np.full((self.size, self.size, 3), self.pad_value, np.uint8)
                square[:h, :w] = img_bgr
                b, s = self._pass(square, 1.0, (0, 0))
                all_b.append(b[s >= self.tile_thresh])
                all_s.append(s[s >= self.tile_thresh])
            else:
                for x0, y0 in tile_origins(h, w, self.size):
                    crop = img_bgr[y0:y0 + self.size, x0:x0 + self.size]
                    square = np.full((self.size, self.size, 3), self.pad_value, np.uint8)
                    square[:crop.shape[0], :crop.shape[1]] = crop
                    b, s = self._pass(square, 1.0, (0, 0), (x0, y0))
                    all_b.append(b[s >= self.tile_thresh])
                    all_s.append(s[s >= self.tile_thresh])
        boxes = np.concatenate(all_b) if all_b else np.zeros((0, 4), np.float32)
        scores = np.concatenate(all_s) if all_s else np.zeros((0,), np.float32)
        keep = nms(boxes, scores, self.nms_thresh)
        boxes, scores = boxes[keep], scores[keep]
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
        return [Detection(*map(float, b), float(s), self.label) for b, s in zip(boxes, scores)]


class FaceDetector(_TiledDetector):
    """YuNet face detector (OpenCV Zoo, MIT). Fixed 640 x 640 input, BGR 0..255."""

    label = "face"
    pad_value = 0

    def __init__(self, model_path: str, score_thresh: float = 0.4, nms_thresh: float = 0.3,
                 tiled: bool = True):
        super().__init__(model_path, score_thresh, nms_thresh, tiled)

    def _infer(self, square):
        blob = square.transpose(2, 0, 1)[None].astype(np.float32)
        outs = dict(zip(self.out_names, self.sess.run(None, {self.input_name: blob})))
        return decode_yunet(outs, self.size, self.score_thresh)


class PlateDetector(_TiledDetector):
    """Number-plate detector.

    Accepts two ONNX output layouts:
    * end-to-end YOLOv9 from open-image-models: rows of
      (batch, x1, y1, x2, y2, class, score)
    * Ultralytics export with NMS: (1, N, 6) rows of (x1, y1, x2, y2, score, class)
    """

    label = "plate"
    pad_value = 114

    def __init__(self, model_path: str, score_thresh: float = 0.2, nms_thresh: float = 0.5,
                 tiled: bool = False, tile_thresh: float | None = None):
        super().__init__(model_path, score_thresh, nms_thresh, tiled, tile_thresh)

    def _infer(self, square):
        blob = square[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        out = self.sess.run(None, {self.input_name: np.ascontiguousarray(blob)})[0]
        boxes, scores = self._parse(out)
        keep = scores >= self.score_thresh
        return boxes[keep], scores[keep]

    @staticmethod
    def _parse(out: np.ndarray):
        if out.ndim == 2 and out.shape[1] == 7:
            return out[:, 1:5].astype(np.float32), out[:, 6].astype(np.float32)
        if out.ndim == 3 and out.shape[2] == 6:
            rows = out[0]
            return rows[:, 0:4].astype(np.float32), rows[:, 4].astype(np.float32)
        raise ValueError(f"Unsupported plate model output shape {out.shape}")
