"""Model registry and download cache.

Weights are not stored in the Python package. They are fetched once from the
upstream projects, checked against a pinned SHA-256 and cached locally.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelSpec:
    name: str
    url: str
    sha256: str
    license: str
    source: str


MODELS: dict[str, ModelSpec] = {
    "face-yunet-640": ModelSpec(
        name="face-yunet-640",
        url=(
            "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
            "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
        ),
        sha256="8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
        license="MIT",
        source="OpenCV Zoo, YuNet (Shiqi Yu et al.)",
    ),
    "plate-yolov9t-384": ModelSpec(
        name="plate-yolov9t-384",
        url=(
            "https://github.com/ankandrew/open-image-models/releases/download/"
            "assets/yolo-v9-t-384-license-plates-end2end.onnx"
        ),
        sha256="888397b96d761c89db40bc9c305838e8652660f5e282c2cadebbe8d2951a77a8",
        license="MIT",
        source="open-image-models (ankandrew)",
    ),
    "plate-yolov9t-640": ModelSpec(
        name="plate-yolov9t-640",
        url=(
            "https://github.com/ankandrew/open-image-models/releases/download/"
            "assets/yolo-v9-t-640-license-plates-end2end.onnx"
        ),
        sha256="c3c1026ca7d0585dd88084d68182dd897113712fa734ae1557ca70174440c076",
        license="MIT",
        source="open-image-models (ankandrew)",
    ),
}

DEFAULT_FACE = "face-yunet-640"
DEFAULT_PLATE = "plate-yolov9t-640"


def cache_dir() -> Path:
    root = os.environ.get("PARDA_CACHE")
    if root:
        return Path(root)
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "parda"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve(name_or_path: str) -> Path:
    """Return a local path for a registered model name or an existing file path."""
    p = Path(name_or_path)
    if p.suffix == ".onnx" and p.exists():
        return p
    if name_or_path not in MODELS:
        raise KeyError(f"Unknown model '{name_or_path}'. Known: {', '.join(MODELS)}")
    spec = MODELS[name_or_path]
    target = cache_dir() / f"{spec.name}.onnx"
    if target.exists() and sha256_of(target) == spec.sha256:
        return target
    fetch(spec, target)
    return target


def fetch(spec: ModelSpec, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, dir=target.parent, suffix=".part") as tmp:
        tmp_path = Path(tmp.name)
        with urllib.request.urlopen(spec.url, timeout=120) as r:  # noqa: S310 (pinned https URL)
            shutil.copyfileobj(r, tmp)
    digest = sha256_of(tmp_path)
    if digest != spec.sha256:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Checksum mismatch for {spec.name}: expected {spec.sha256}, got {digest}"
        )
    tmp_path.replace(target)
