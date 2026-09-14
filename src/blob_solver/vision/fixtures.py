"""Deterministic fixture-image generation for developer tests and demos."""

from __future__ import annotations

from typing import Any


FIXTURE_PALETTE = {
    "red": (251, 88, 93),
    "yellow": (251, 199, 57),
    "green": (136, 198, 37),
    "blue": (25, 128, 193),
    None: (16, 29, 61),
}


def make_board_image(matrix: list[list[str | None]], *, cell_size: int = 40) -> Any:
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise RuntimeError("Pillow is required for fixture generation") from exc
    if not matrix or not matrix[0]:
        raise ValueError("fixture matrix must be non-empty")
    width = len(matrix[0])
    if any(len(row) != width for row in matrix):
        raise ValueError("fixture matrix must be rectangular")
    image = Image.new(
        "RGB",
        (width * cell_size, len(matrix) * cell_size),
        FIXTURE_PALETTE[None],
    )
    draw = ImageDraw.Draw(image)
    for row, values in enumerate(matrix):
        for col, symbol in enumerate(values):
            if symbol is None:
                continue
            left, top = col * cell_size, row * cell_size
            draw.rounded_rectangle(
                (left + 4, top + 4, left + cell_size - 4, top + cell_size - 4),
                radius=max(2, cell_size // 8),
                fill=FIXTURE_PALETTE[symbol],
            )
    return image
