"""Laika image processing engine."""

from laika.depth import DepthAnythingV2Small, DepthInferenceError, DepthPrediction
from laika.m9 import M9ColorEngine, M9ColorResult, M9InferenceError, M9Onnx
from laika.processing import (
    LaikaProcessingEngine,
    ProcessingCancelled,
    ProcessingError,
    ProcessingOptions,
    ProcessingProgress,
    ProcessingResult,
)

__version__ = "0.1.0"

__all__ = [
    "DepthAnythingV2Small",
    "DepthInferenceError",
    "DepthPrediction",
    "M9ColorEngine",
    "M9ColorResult",
    "M9InferenceError",
    "M9Onnx",
    "LaikaProcessingEngine",
    "ProcessingCancelled",
    "ProcessingError",
    "ProcessingOptions",
    "ProcessingProgress",
    "ProcessingResult",
    "__version__",
]
