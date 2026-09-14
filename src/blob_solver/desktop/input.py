"""Shared safety helpers for desktop input implementations."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Event

from .base import DesktopUnavailable, InputBackend


@dataclass(slots=True)
class SafeClicker:
    """Wrap an input backend with a cooperative emergency-stop latch."""

    backend: InputBackend
    stop_event: Event

    def click(self, x: int, y: int) -> None:
        if self.stop_event.is_set():
            raise DesktopUnavailable("click suppressed by emergency stop")
        self.backend.click(x, y)

    def emergency_stop(self) -> None:
        self.stop_event.set()
        self.backend.emergency_stop()
