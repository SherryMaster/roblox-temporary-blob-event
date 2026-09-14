"""Human-readable and graphical views of vision classifications."""

from __future__ import annotations

from typing import Any

from .classifier import BoardObservation


DEBUG_COLORS = {
    "red": (251, 88, 93),
    "yellow": (251, 199, 57),
    "green": (136, 198, 37),
    "blue": (25, 128, 193),
    "?": (220, 60, 220),
    None: (16, 29, 61),
}


def format_observation(observation: BoardObservation) -> str:
    lines = [
        observation.text(),
        "",
        f"valid={observation.valid} min_confidence={observation.min_confidence:.3f}",
    ]
    if observation.error:
        lines.append(f"error={observation.error}")
    uncertain = [
        f"({cell.row},{cell.col})={cell.confidence:.3f}"
        for cell in observation.low_confidence_cells
    ]
    if uncertain:
        lines.append("uncertain: " + ", ".join(uncertain))
    return "\n".join(lines)


def render_observation(
    observation: BoardObservation,
    *,
    cell_size: int = 72,
    image_module: Any | None = None,
) -> Any:
    """Render labels and confidence values into a board-debug image."""

    if image_module is None:
        try:
            from PIL import Image, ImageDraw
        except ImportError as exc:
            raise RuntimeError("Pillow is required for graphical vision debug") from exc
    else:
        Image = image_module.Image
        ImageDraw = image_module.ImageDraw
    image = Image.new(
        "RGB",
        (observation.cols * cell_size, observation.rows * cell_size),
        DEBUG_COLORS[None],
    )
    draw = ImageDraw.Draw(image)
    for cell in observation.cells:
        left, top = cell.col * cell_size, cell.row * cell_size
        color = DEBUG_COLORS.get(cell.symbol, (128, 128, 128))
        draw.rectangle(
            (left + 2, top + 2, left + cell_size - 2, top + cell_size - 2),
            fill=color,
            outline=(245, 245, 245),
        )
        label = "." if cell.symbol is None else cell.symbol[0].upper() if cell.symbol != "?" else "?"
        draw.text((left + 7, top + 7), label, fill=(20, 20, 25))
        draw.text((left + 7, top + cell_size - 20), f"{cell.confidence:.2f}", fill=(245, 245, 245))
        if cell.confidence < observation.confidence_threshold:
            draw.rectangle(
                (left + 1, top + 1, left + cell_size - 1, top + cell_size - 1),
                outline=(255, 30, 30),
                width=4,
            )
    return image
