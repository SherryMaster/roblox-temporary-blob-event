"""Interfaces separating desktop effects from game logic and vision."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, Protocol

from blob_solver.vision.region import Region


class DesktopUnavailable(RuntimeError):
    pass


class CaptureBackend(Protocol):
    def capture(self, region: Region) -> Any:
        ...


class RegionSelector(Protocol):
    def select_region(self) -> Region:
        ...


class InputBackend(Protocol):
    def click(self, x: int, y: int) -> None:
        ...

    def emergency_stop(self) -> None:
        ...


class OverlayBackend(Protocol):
    def show(self, spec: Any) -> None:
        ...

    def hide(self) -> None:
        ...

    def close(self) -> None:
        ...

    @contextmanager
    def hidden_during_capture(self) -> Iterator[None]:
        self.hide()
        try:
            yield
        finally:
            pass


class DesktopBackend(Protocol):
    capture: CaptureBackend
    selector: RegionSelector
    input: InputBackend
    overlay: OverlayBackend
