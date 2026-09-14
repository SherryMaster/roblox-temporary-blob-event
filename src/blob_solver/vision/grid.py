"""Regular-grid extraction using robust inner-patch RGB samples."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Any

from .region import Region


@dataclass(frozen=True, slots=True)
class GridSpec:
    rows: int = 10
    cols: int = 10

    def __post_init__(self) -> None:
        if self.rows < 1 or self.cols < 1:
            raise ValueError("grid rows and cols must be positive")


@dataclass(frozen=True, slots=True)
class CellSample:
    row: int
    col: int
    rgb: tuple[int, int, int]
    patch_box: tuple[int, int, int, int]


def _rgb_pixels(image: Any, box: tuple[int, int, int, int]) -> list[tuple[int, int, int]]:
    patch = image.crop(box).convert("RGB")
    pixels = patch.get_flattened_data() if hasattr(patch, "get_flattened_data") else patch.getdata()
    return [tuple(int(channel) for channel in pixel) for pixel in pixels]


def robust_rgb(image: Any, box: tuple[int, int, int, int]) -> tuple[int, int, int]:
    """Use per-channel medians so a pointer/outline cannot dominate one sample."""

    pixels = _rgb_pixels(image, box)
    if not pixels:
        raise ValueError("cell patch has no pixels")
    return tuple(int(round(median(channel))) for channel in zip(*pixels))  # type: ignore[arg-type]


def sample_grid(
    image: Any,
    region: Region,
    spec: GridSpec,
    *,
    patch_ratio: float = 0.45,
) -> tuple[CellSample, ...]:
    """Sample a centered inner patch from each logical cell.

    The input image is expected to be the crop corresponding to ``region``.
    Keeping the image/region contract explicit prevents accidental double offsets
    when the capture backend returns a region-sized screenshot.
    """

    if not 0.05 <= patch_ratio <= 1.0:
        raise ValueError("patch_ratio must be between 0.05 and 1.0")
    image_width, image_height = image.size
    if image_width < region.width or image_height < region.height:
        raise ValueError(
            f"capture image {image_width}x{image_height} is smaller than region "
            f"{region.width}x{region.height}"
        )
    if region.width < spec.cols or region.height < spec.rows:
        raise ValueError("region must be at least one pixel wide/high per logical cell")
    samples: list[CellSample] = []
    for row in range(spec.rows):
        for col in range(spec.cols):
            left, top, right, bottom = region.cell_rect(spec.rows, spec.cols, row, col)
            cell_width, cell_height = right - left, bottom - top
            patch_width = max(1, int(round(cell_width * patch_ratio)))
            patch_height = max(1, int(round(cell_height * patch_ratio)))
            patch_left = left + max(0, (cell_width - patch_width) // 2)
            patch_top = top + max(0, (cell_height - patch_height) // 2)
            patch_box = (
                patch_left,
                patch_top,
                min(right, patch_left + patch_width),
                min(bottom, patch_top + patch_height),
            )
            samples.append(CellSample(row=row, col=col, rgb=robust_rgb(image, patch_box), patch_box=patch_box))
    return tuple(samples)
