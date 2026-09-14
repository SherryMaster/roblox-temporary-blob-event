"""Geometry-aware board capture, calibration, and classification."""

from .calibration import CalibrationProfile, calibrate
from .capture import MemoryCapture, ScreenCapture, WaylandCapture, capture_backend
from .classifier import BoardObservation, CellClassification, VisionClassifier, classify_samples
from .grid import CellSample, GridSpec, sample_grid
from .region import Region
from .settle import BoardSettler, SettledObservation, SettleTimeout

__all__ = [
    "BoardSettler",
    "BoardObservation",
    "CellClassification",
    "CalibrationProfile",
    "CellSample",
    "GridSpec",
    "MemoryCapture",
    "Region",
    "ScreenCapture",
    "SettleTimeout",
    "SettledObservation",
    "WaylandCapture",
    "VisionClassifier",
    "calibrate",
    "classify_samples",
    "capture_backend",
    "sample_grid",
]
