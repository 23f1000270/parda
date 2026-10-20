import numpy as np
import pytest

from parda.detect import Detection, PlateDetector, decode_yunet, nms, tile_origins
from parda.mask import apply_masks, expand, full_frame
from parda.metrics import coverage, evaluate, mask_raster
from parda.pipeline import Anonymizer, Config
from parda.track import MaskMemory


# ---------- masking ----------

def test_expand_clips_to_frame():
    assert expand((0, 0, 10, 10), 0.5, 12, 12) == (0, 0, 12, 12)
    assert expand((10, 10, 20, 20), 0.1, 100, 100) == (9, 9, 21, 21)


def test_masks_only_touch_boxes():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (60, 80, 3), dtype=np.uint8)
    out = apply_masks(img, [(10, 10, 30, 40)], rng=rng)
    outside = np.ones(img.shape[:2], bool)
    outside[10:40, 10:30] = False
    assert np.array_equal(out[outside], img[outside])
    assert not np.array_equal(out[10:40, 10:30], img[10:40, 10:30])


def test_blocks_leave_only_a_coarse_grid():
    rng = np.random.default_rng(1)
    img = rng.integers(0, 255, (120, 120, 3), dtype=np.uint8)
    out = apply_masks(img, [(0, 0, 120, 120)], blocks=6, rng=rng)
    # 6 x 6 cells of 20 px, each a single flat colour
    for y in range(0, 120, 20):
        for x in range(0, 120, 20):
            cell = out[y:y + 20, x:x + 20].reshape(-1, 3)
            assert (cell == cell[0]).all()


def test_full_frame_hides_everything():
    img = np.full((10, 10, 3), 200, np.uint8)
    assert full_frame(img).max() < 50


# ---------- detection helpers ----------

def test_tiles_cover_frame():
    origins = tile_origins(1080, 1920, 640)
    cover = np.zeros((1080, 1920), bool)
    for x, y in origins:
        cover[y:y + 640, x:x + 640] = True
    assert cover.all()
    assert tile_origins(500, 600, 640) == [(0, 0)]


def test_nms_suppresses_overlaps():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], np.float32)
    keep = nms(boxes, np.array([0.9, 0.8, 0.7], np.float32), 0.5)
    assert sorted(keep.tolist()) == [0, 2]


def test_decode_yunet_single_cell():
    size = 640
    outs = {}
    for s in (8, 16, 32):
        n = (size // s) ** 2
        outs[f"cls_{s}"] = np.zeros((1, n, 1), np.float32)
        outs[f"obj_{s}"] = np.zeros((1, n, 1), np.float32)
        outs[f"bbox_{s}"] = np.zeros((1, n, 4), np.float32)
    cols = size // 16
    idx = 3 * cols + 5  # row 3, col 5 at stride 16
    outs["cls_16"][0, idx, 0] = 1.0
    outs["obj_16"][0, idx, 0] = 1.0
    outs["bbox_16"][0, idx] = [0.5, 0.5, np.log(2.0), np.log(3.0)]
    boxes, scores = decode_yunet(outs, size, 0.5)
    assert len(boxes) == 1 and scores[0] == pytest.approx(1.0)
    cx, cy, w, h = (5.5) * 16, (3.5) * 16, 32, 48
    assert boxes[0] == pytest.approx([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


def test_plate_output_layouts():
    e2e = np.array([[0, 1, 2, 3, 4, 0, 0.9]], np.float32)
    b, s = PlateDetector._parse(e2e)
    assert b.tolist() == [[1, 2, 3, 4]] and s.tolist() == pytest.approx([0.9])
    ultra = np.array([[[1, 2, 3, 4, 0.8, 0]]], np.float32)
    b, s = PlateDetector._parse(ultra)
    assert b.tolist() == [[1, 2, 3, 4]] and s.tolist() == pytest.approx([0.8])


# ---------- mask memory ----------

def _det(x, label="face"):
    return Detection(x, 10, x + 20, 30, 0.9, label)


def test_memory_holds_lost_objects_and_grows_them():
    mem = MaskMemory(hold_frames=3)
    first = mem.update([_det(10)])
    assert len(first) == 1
    held = mem.update([])  # detector flickers
    assert len(held) == 1
    w0 = first[0][0][2] - first[0][0][0]
    w1 = held[0][0][2] - held[0][0][0]
    assert w1 > w0
    for _ in range(3):
        mem.update([])
    assert mem.update([]) == []


def test_memory_follows_motion():
    mem = MaskMemory(hold_frames=5)
    mem.update([_det(10)])
    mem.update([_det(20)])
    predicted = mem.update([])[0][0]
    assert (predicted[0] + predicted[2]) / 2 > 30  # kept moving right


# ---------- metrics ----------

def test_coverage_and_over_mask():
    r = mask_raster([(0, 0, 10, 10)], 20, 20)
    assert coverage((0, 0, 10, 10), r) == 1.0
    assert coverage((0, 0, 20, 10), r) == pytest.approx(0.5)
    res = evaluate([(20, 20, [("face", (0, 0, 10, 10), {"light": "day"})], [(0, 0, 10, 20)])])
    assert res.by_group["face"].recall(0.95) == 1.0
    assert res.by_group["face / light=day"].total == 1
    assert res.over_mask == pytest.approx(100 / 400)


# ---------- pipeline ----------

class _Fake:
    def __init__(self, dets=None, boom=False):
        self.dets, self.boom = dets or [], boom

    def detect(self, img):
        if self.boom:
            raise RuntimeError("model crashed")
        return self.dets


def test_pipeline_masks_detections():
    img = np.random.default_rng(3).integers(0, 255, (100, 100, 3), dtype=np.uint8)
    a = Anonymizer(Config(), seed=0, face_detector=_Fake([Detection(40, 40, 60, 60, 0.9, "face")]),
                   plate_detector=_Fake())
    res = a.process(img)
    assert res.failsafe is None and len(res.masked_boxes) == 1
    x1, y1, x2, y2 = res.masked_boxes[0]
    assert x1 < 40 and y1 < 40 and x2 > 60 and y2 > 60  # margin added
    assert not np.array_equal(res.image[40:60, 40:60], img[40:60, 40:60])
    assert np.array_equal(res.image[:30, :30], img[:30, :30])


def test_pipeline_fails_closed():
    img = np.full((50, 50, 3), 250, np.uint8)
    a = Anonymizer(Config(), face_detector=_Fake(boom=True), plate_detector=_Fake())
    res = a.process(img)
    assert res.failsafe and res.image.max() < 50
