"""Desktop backend selection based on explicit config or session environment."""

from __future__ import annotations

import os

from .base import DesktopUnavailable
from .overlay import create_overlay
from .wayland import WaylandDesktop
from .x11 import X11Desktop


def create_desktop_backend(name: str = "auto"):
    name = name.lower()
    if name not in {"auto", "wayland", "x11"}:
        raise ValueError("desktop backend must be auto, wayland, or x11")
    wayland_error: Exception | None = None
    if name in {"auto", "wayland"} and os.environ.get("WAYLAND_DISPLAY"):
        try:
            backend = WaylandDesktop()
            backend.overlay = create_overlay()
            return backend
        except Exception as exc:
            wayland_error = exc
            if name == "wayland":
                raise
    if name == "auto" and os.environ.get("WAYLAND_DISPLAY") and not os.environ.get("DISPLAY"):
        raise DesktopUnavailable(f"Wayland backend unavailable: {wayland_error}")
    if name in {"auto", "x11"}:
        backend = X11Desktop()
        backend.overlay = create_overlay()
        return backend
    raise DesktopUnavailable("no supported desktop session detected")
