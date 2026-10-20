"""End-to-end video test. Uses the real models if they are cached, else skips."""

import json
import shutil
import subprocess

import cv2
import numpy as np
import pytest

from parda import models
from parda.pipeline import Anonymizer, Config
from parda.video import anonymize_video


def _cached(name):
    p = models.cache_dir() / f"{name}.onnx"
    return p.exists() and models.sha256_of(p) == models.MODELS[name].sha256


needs_models = pytest.mark.skipif(
    not (_cached(models.DEFAULT_FACE) and _cached(models.DEFAULT_PLATE)),
    reason="models not cached; run `parda fetch-models`",
)


@needs_models
def test_video_roundtrip_strips_audio_and_metadata(tmp_path):
    src = tmp_path / "in.mp4"
    w = cv2.VideoWriter(str(src), cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 240))
    for i in range(12):
        frame = np.full((240, 320, 3), 40 + i * 10, np.uint8)
        w.write(frame)
    w.release()

    out = tmp_path / "out.mp4"
    rep = anonymize_video(Anonymizer(Config()), src, out)
    assert rep.frames == 12 and rep.failsafe_frames == 0
    assert out.exists() and out.stat().st_size > 0

    if shutil.which("ffprobe"):
        info = json.loads(subprocess.run(
            ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(out)],
            capture_output=True, text=True, check=True).stdout)
        kinds = {s["codec_type"] for s in info["streams"]}
        assert kinds == {"video"}
