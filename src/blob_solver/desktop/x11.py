"""Optional X11 fallback using Pillow/mss capture and xdotool input."""

from __future__ import annotations

import shutil
import subprocess

from blob_solver.vision.capture import MSSCapture, PillowCapture
from blob_solver.vision.region import Region

from .base import DesktopUnavailable


class X11RegionSelector:
    def __init__(self, command: str = "slop") -> None:
        self.command = command

    def select_region(self) -> Region:
        if shutil.which(self.command) is None:
            raise DesktopUnavailable("install slop or use --region x,y,width,height for X11")
        completed = subprocess.run(
            [self.command, "-f", "%x,%y %w %h"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode != 0:
            raise DesktopUnavailable(completed.stderr.strip() or "slop failed")
        return Region.parse(completed.stdout)


class XdotoolInput:
    def __init__(self, command: str = "xdotool") -> None:
        if shutil.which(command) is None:
            raise DesktopUnavailable("xdotool is required for X11 autoplay")
        self.command = command
        self._stopped = False

    def click(self, x: int, y: int) -> None:
        if self._stopped:
            raise DesktopUnavailable("input backend is stopped")
        completed = subprocess.run(
            [self.command, "mousemove", str(x), str(y), "click", "1"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if completed.returncode != 0:
            raise DesktopUnavailable(completed.stderr.decode(errors="replace").strip() or "xdotool failed")

    def emergency_stop(self) -> None:
        self._stopped = True


class UnavailableInput:
    def click(self, x: int, y: int) -> None:
        raise DesktopUnavailable("xdotool is unavailable; X11 hint/scan mode can run without autoplay")

    def emergency_stop(self) -> None:
        pass


class X11Desktop:
    def __init__(self, *, capture: str = "auto") -> None:
        if capture == "mss":
            self.capture = MSSCapture()
        elif capture == "pillow":
            self.capture = PillowCapture()
        else:
            try:
                self.capture = MSSCapture()
            except Exception:
                self.capture = PillowCapture()
        self.selector = X11RegionSelector()
        try:
            self.input = XdotoolInput()
        except DesktopUnavailable:
            self.input = UnavailableInput()
        self.overlay = None
