"""Parda: on-device anonymisation of faces and number plates for Indian road video."""

from .detect import Detection
from .pipeline import Anonymizer, Config

__all__ = ["Anonymizer", "Config", "Detection"]
__version__ = "0.1.0"
