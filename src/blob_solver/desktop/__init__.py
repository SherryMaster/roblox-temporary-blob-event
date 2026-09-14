"""Platform-specific capture, input, and hint overlay adapters."""

from .base import DesktopBackend, DesktopUnavailable, InputBackend, OverlayBackend
from .factory import create_desktop_backend

__all__ = [
    "DesktopBackend",
    "DesktopUnavailable",
    "InputBackend",
    "OverlayBackend",
    "create_desktop_backend",
]
