"""Screen capture interfaces and grim/Pillow implementations."""

from __future__ import annotations

from io import BytesIO
import os
import shutil
import subprocess
from typing import Any, Protocol

from .region import Region


class ScreenCapture(Protocol):
    def capture(self, region: Region) -> Any:
        """Return a Pillow-compatible RGB image cropped to ``region``."""


class CaptureError(RuntimeError):
    pass


def _load_pillow() -> Any:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - exercised in minimal installs
        raise CaptureError("Pillow is required for screen capture; install blob-solver") from exc
    return Image


class WaylandCapture:
    """Capture an exact region with ``grim`` under Wayland."""

    def __init__(self, command: str = "grim") -> None:
        if shutil.which(command) is None:
            raise CaptureError(f"{command!r} was not found on PATH")
        self.command = command

    def capture(self, region: Region) -> Any:
        completed = subprocess.run(
            [self.command, "-g", region.to_grim_geometry(), "-"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if completed.returncode != 0:
            detail = completed.stderr.decode(errors="replace").strip()
            raise CaptureError(f"grim failed: {detail or completed.returncode}")
        image_module = _load_pillow()
        with image_module.open(BytesIO(completed.stdout)) as image:
            return image.convert("RGB")


class PillowCapture:
    """X11-friendly fallback using Pillow's ImageGrab."""

    def capture(self, region: Region) -> Any:
        image_module = _load_pillow()
        try:
            image = image_module.grab(bbox=(region.x, region.y, region.x + region.width, region.y + region.height))
        except Exception as exc:  # pragma: no cover - desktop dependent
            raise CaptureError(f"Pillow ImageGrab failed: {exc}") from exc
        return image.convert("RGB")


class MSSCapture:
    """Optional X11 capture with mss when ImageGrab is unavailable."""

    def __init__(self) -> None:
        try:
            import mss
        except ImportError as exc:  # pragma: no cover - optional backend
            raise CaptureError("mss is not installed") from exc
        self._mss = mss.mss()

    def capture(self, region: Region) -> Any:
        image_module = _load_pillow()
        shot = self._mss.grab({"left": region.x, "top": region.y, "width": region.width, "height": region.height})
        return image_module.frombytes("RGB", shot.size, shot.rgb)


class MemoryCapture:
    """Test/developer capture backend returning a supplied Pillow image."""

    def __init__(self, image: Any) -> None:
        self.image = image
        self.capture_count = 0

    def capture(self, region: Region) -> Any:
        self.capture_count += 1
        if self.image.size[0] < region.width or self.image.size[1] < region.height:
            raise CaptureError("memory image is smaller than requested region")
        return self.image.copy()


def capture_backend(name: str = "auto") -> ScreenCapture:
    """Select a backend without importing desktop-only libraries eagerly."""

    name = name.lower()
    if name not in {"auto", "wayland", "x11", "pillow", "mss"}:
        raise ValueError(f"unknown capture backend: {name}")
    if name in {"auto", "wayland"} and os.environ.get("WAYLAND_DISPLAY"):
        try:
            return WaylandCapture()
        except CaptureError:
            if name == "wayland":
                raise
    if name == "mss":
        return MSSCapture()
    if name in {"x11", "pillow"}:
        return PillowCapture()
    try:
        return MSSCapture()
    except CaptureError:
        return PillowCapture()
