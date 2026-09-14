"""Desktop backend selection based on explicit config or session environment."""

from __future__ import annotations

import os

from .base import DesktopUnavailable
from .overlay import create_overlay, create_wayland_overlay
from .wayland import WaylandDesktop
from .x11 import X11Desktop


def create_desktop_backend(name: str = "auto"):
    name = name.lower()
    if name not in {"auto", "wayland", "x11"}:
        raise ValueError("desktop backend must be auto, wayland, or x11")
    if name in {"auto", "wayland"} and os.environ.get("WAYLAND_DISPLAY"):
        try:
            backend = WaylandDesktop()
            backend.overlay = create_wayland_overlay()
            return backend
        except Exception as exc:
            # A live Wayland session must not silently become an X11/Qt
            # overlay merely because one Wayland dependency is missing. X11
            # remains available through the explicit ``backend = "x11"``.
            raise DesktopUnavailable(f"Wayland backend unavailable: {exc}") from exc
    if name in {"auto", "x11"}:
        backend = X11Desktop()
        backend.overlay = create_overlay()
        return backend
    raise DesktopUnavailable("no supported desktop session detected")
