"""Laika image processing engine."""

from laika.depth import DepthAnythingV2Small, DepthInferenceError, DepthPrediction

__version__ = "0.1.0"

__all__ = [
    "DepthAnythingV2Small",
    "DepthInferenceError",
    "DepthPrediction",
    "__version__",
]
