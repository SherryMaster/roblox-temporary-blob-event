"""Wayland/Hyprland adapters using grim, slurp, and ydotool."""

from __future__ import annotations

import shutil
import subprocess
from typing import Any

from blob_solver.vision.capture import WaylandCapture
from blob_solver.vision.region import Region

from .base import DesktopUnavailable


def _require(command: str) -> None:
    if shutil.which(command) is None:
        raise DesktopUnavailable(f"{command!r} is required for this Wayland backend")


class WaylandRegionSelector:
    def __init__(self, command: str = "slurp") -> None:
        _require(command)
        self.command = command

    def select_region(self) -> Region:
        completed = subprocess.run(
            [self.command, "-f", "%x,%y %wx%h"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip()
            raise DesktopUnavailable(f"slurp failed: {detail or completed.returncode}")
        try:
            return Region.parse(completed.stdout)
        except ValueError as exc:
            raise DesktopUnavailable(f"slurp returned an invalid region: {completed.stdout!r}") from exc


class YdotoolInput:
    """Absolute pointer/click input through ydotool's uinput daemon."""

    def __init__(self, command: str = "ydotool") -> None:
        _require(command)
        self.command = command
        self._stopped = False

    def click(self, x: int, y: int) -> None:
        if self._stopped:
            raise DesktopUnavailable("input backend is stopped; create a new backend to resume")
        move = subprocess.run(
            [self.command, "mousemove", "--absolute", "-x", str(x), "-y", str(y)],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if move.returncode != 0:
            detail = move.stderr.decode(errors="replace").strip()
            raise DesktopUnavailable(detail or "ydotool mousemove failed")
        click = subprocess.run(
            [self.command, "click", "0xC0"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if click.returncode != 0:
            detail = click.stderr.decode(errors="replace").strip()
            raise DesktopUnavailable(detail or "ydotool click failed")

    def emergency_stop(self) -> None:
        self._stopped = True


class UnavailableInput:
    def click(self, x: int, y: int) -> None:
        raise DesktopUnavailable("ydotool is unavailable; install/configure ydotoold for autoplay")

    def emergency_stop(self) -> None:
        pass


class WaylandDesktop:
    def __init__(self) -> None:
        self.capture = WaylandCapture()
        self.selector = WaylandRegionSelector()
        try:
            self.input: Any = YdotoolInput()
        except DesktopUnavailable:
            self.input = UnavailableInput()
        self.overlay = None


def select_wayland_region() -> Region:
    return WaylandRegionSelector().select_region()
