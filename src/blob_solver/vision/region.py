"""Screen-region and grid geometry value objects."""

from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True, slots=True)
class Region:
    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("region width and height must be positive")

    @classmethod
    def parse(cls, value: str) -> "Region":
        """Parse ``x,y,width,height`` or slurp's ``x,y widthxheight``."""

        text = value.strip()
        comma_form = re.fullmatch(
            r"\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*",
            text,
        )
        if comma_form:
            return cls(*(int(part) for part in comma_form.groups()))
        slurp_form = re.fullmatch(
            r"\s*(-?\d+)\s*,\s*(-?\d+)\s+(\d+)(?:x|\s+)(\d+)\s*",
            text,
        )
        if slurp_form:
            return cls(*(int(part) for part in slurp_form.groups()))
        raise ValueError("region must look like x,y,width,height or x,y widthxheight")

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "Region":
        return cls(int(value["x"]), int(value["y"]), int(value["width"]), int(value["height"]))

    def to_dict(self) -> dict[str, int]:
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}

    def to_grim_geometry(self) -> str:
        return f"{self.x},{self.y} {self.width}x{self.height}"

    def cell_rect(self, rows: int, cols: int, row: int, col: int) -> tuple[int, int, int, int]:
        """Return a cell rectangle in screen coordinates relative to this region."""

        if not (0 <= row < rows and 0 <= col < cols):
            raise IndexError("cell outside grid")
        left = round(col * self.width / cols)
        right = round((col + 1) * self.width / cols)
        top = round(row * self.height / rows)
        bottom = round((row + 1) * self.height / rows)
        return left, top, right, bottom

    def cell_center(self, rows: int, cols: int, row: int, col: int) -> tuple[int, int]:
        left, top, right, bottom = self.cell_rect(rows, cols, row, col)
        return self.x + (left + right) // 2, self.y + (top + bottom) // 2
