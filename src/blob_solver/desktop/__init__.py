"""Platform-specific capture, input, and hint overlay adapters."""

from .base import DesktopBackend, DesktopUnavailable, InputBackend, OverlayBackend
from .coordinates import CoordinateDiagnostic, CoordinateMapper, MonitorInfo
from .factory import create_desktop_backend

__all__ = [
    "DesktopBackend",
    "DesktopUnavailable",
    "CoordinateDiagnostic",
    "CoordinateMapper",
    "InputBackend",
    "MonitorInfo",
    "OverlayBackend",
    "create_desktop_backend",
]
