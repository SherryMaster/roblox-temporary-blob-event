"""Explicit Hyprland/output coordinate conversions for Wayland automation."""

from __future__ import annotations

from dataclasses import dataclass
import json
import shutil
import subprocess
from typing import Any, Iterable

from blob_solver.vision.region import Region


@dataclass(frozen=True, slots=True)
class MonitorInfo:
    """Hyprland monitor geometry in compositor layout coordinates.

    Hyprland reports the mode ``width``/``height`` in physical pixels while
    ``x``/``y`` are layout positions. Layer-shell and grim use the monitor's
    logical size, so the logical dimensions are derived with ``scale``.
    """

    name: str
    x: int
    y: int
    physical_width: int
    physical_height: int
    scale: float
    focused: bool = False
    transform: int = 0

    @property
    def logical_width(self) -> int:
        size = round(self.physical_width / self.scale)
        return max(1, round(self.physical_height / self.scale) if self.transform in {1, 3, 5, 7} else size)

    @property
    def logical_height(self) -> int:
        size = round(self.physical_height / self.scale)
        return max(1, round(self.physical_width / self.scale) if self.transform in {1, 3, 5, 7} else size)

    @property
    def logical_region(self) -> Region:
        return Region(self.x, self.y, self.logical_width, self.logical_height)

    def contains(self, region: Region) -> bool:
        return (
            region.x >= self.x
            and region.y >= self.y
            and region.x + region.width <= self.x + self.logical_width
            and region.y + region.height <= self.y + self.logical_height
        )


class CoordinateError(RuntimeError):
    pass


def _json_payload(output: str) -> Any:
    """Parse hyprctl output even when an older build prefixes diagnostics."""

    start = min((index for index in (output.find("["), output.find("{")) if index >= 0), default=-1)
    if start < 0:
        raise CoordinateError("hyprctl did not return JSON")
    try:
        return json.loads(output[start:])
    except json.JSONDecodeError as exc:
        raise CoordinateError(f"invalid hyprctl monitor JSON: {exc}") from exc


def monitor_info_from_json(value: dict[str, Any]) -> MonitorInfo:
    scale = float(value.get("scale", 1.0))
    if scale <= 0:
        raise CoordinateError("Hyprland returned a non-positive monitor scale")
    return MonitorInfo(
        name=str(value["name"]),
        x=int(value["x"]),
        y=int(value["y"]),
        physical_width=int(value["width"]),
        physical_height=int(value["height"]),
        scale=scale,
        focused=bool(value.get("focused", False)),
        transform=int(value.get("transform", 0)),
    )


def query_hyprland_monitors(command: str = "hyprctl") -> tuple[MonitorInfo, ...]:
    if shutil.which(command) is None:
        raise CoordinateError(f"{command!r} is required for Hyprland coordinate diagnostics")
    completed = subprocess.run(
        [command, "monitors", "-j"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        raise CoordinateError(completed.stderr.strip() or "hyprctl monitors failed")
    payload = _json_payload(completed.stdout)
    if not isinstance(payload, list):
        raise CoordinateError("hyprctl monitors JSON was not an array")
    return tuple(monitor_info_from_json(item) for item in payload if isinstance(item, dict))


@dataclass(frozen=True, slots=True)
class OverlayGeometry:
    monitor: MonitorInfo
    region: Region
    local_region: Region


@dataclass(frozen=True, slots=True)
class CoordinateDiagnostic:
    capture_region: Region
    monitor: str
    scale: float
    logical_overlay_region: Region
    logical_click_point: tuple[int, int]
    input_click_point: tuple[int, int]
    input_space: str

    def as_dict(self) -> dict[str, object]:
        return {
            "capture_region": self.capture_region.to_dict(),
            "monitor": self.monitor,
            "scale": self.scale,
            "logical_overlay_region": self.logical_overlay_region.to_dict(),
            "logical_click_point": list(self.logical_click_point),
            "input_click_point": list(self.input_click_point),
            "input_space": self.input_space,
        }


class CoordinateMapper:
    """Convert one selected layout region to layer and input coordinates."""

    def __init__(
        self,
        monitors: Iterable[MonitorInfo] | None = None,
        *,
        input_space: str = "logical",
    ) -> None:
        if input_space not in {"logical", "physical"}:
            raise ValueError("input_space must be logical or physical")
        self.monitors = tuple(monitors or ())
        self.input_space = input_space

    def monitor_for(self, region: Region) -> MonitorInfo:
        containing = tuple(monitor for monitor in self.monitors if monitor.contains(region))
        if len(containing) == 1:
            return containing[0]
        if not containing:
            raise CoordinateError(
                "selected region is not wholly inside one Hyprland monitor; "
                "select a single-output board region"
            )
        raise CoordinateError("selected region matched multiple monitor geometries")

    def overlay_geometry(self, region: Region) -> OverlayGeometry:
        monitor = self.monitor_for(region)
        return OverlayGeometry(
            monitor=monitor,
            region=region,
            local_region=Region(region.x - monitor.x, region.y - monitor.y, region.width, region.height),
        )

    def logical_to_input(self, point: tuple[int, int], monitor: MonitorInfo) -> tuple[int, int]:
        if self.input_space == "logical":
            return point
        # This mode is explicit because uinput absolute axes are commonly
        # physical-pixel based. It is useful for backends that document that
        # contract; the default ydotool path stays in the compositor's logical
        # layout space and the diagnostic prints both values.
        return round(point[0] * monitor.scale), round(point[1] * monitor.scale)

    def click_diagnostic(
        self,
        region: Region,
        *,
        rows: int,
        cols: int,
        screen_row: int,
        col: int,
    ) -> CoordinateDiagnostic:
        geometry = self.overlay_geometry(region)
        logical_point = region.cell_center(rows, cols, screen_row, col)
        return CoordinateDiagnostic(
            capture_region=region,
            monitor=geometry.monitor.name,
            scale=geometry.monitor.scale,
            logical_overlay_region=geometry.local_region,
            logical_click_point=logical_point,
            input_click_point=self.logical_to_input(logical_point, geometry.monitor),
            input_space=self.input_space,
        )
