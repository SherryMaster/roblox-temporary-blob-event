from __future__ import annotations

from PIL import Image, ImageDraw

from blob_solver.vision.calibration import calibrate
from blob_solver.vision.classifier import classify_samples
from blob_solver.vision.grid import GridSpec, sample_grid
from blob_solver.vision.grid import CellSample
from blob_solver.vision.region import Region


PALETTE = {
    "red": (251, 88, 93),
    "yellow": (251, 199, 57),
    "green": (136, 198, 37),
    "blue": (25, 128, 193),
    None: (16, 29, 61),
}


def render_board(matrix: list[list[str | None]], *, cell_size: int = 30) -> Image.Image:
    image = Image.new("RGB", (len(matrix[0]) * cell_size, len(matrix) * cell_size), PALETTE[None])
    draw = ImageDraw.Draw(image)
    for row, values in enumerate(matrix):
        for col, symbol in enumerate(values):
            if symbol is not None:
                left, top = col * cell_size, row * cell_size
                draw.rounded_rectangle(
                    (left + 3, top + 3, left + cell_size - 3, top + cell_size - 3),
                    radius=5,
                    fill=PALETTE[symbol],
                )
    return image


def test_full_board_calibration_and_reconstruction() -> None:
    matrix = [
        ["red", "yellow", "green", "blue"],
        ["yellow", "green", "blue", "red"],
        ["green", "blue", "red", "yellow"],
    ]
    image = render_board(matrix)
    spec = GridSpec(rows=3, cols=4)
    samples = sample_grid(image, Region(0, 0, image.width, image.height), spec)
    profile = calibrate(samples, num_colors=4)
    observation = classify_samples(samples, spec, profile, confidence_threshold=0.85)
    assert observation.valid
    assert observation.matrix == tuple(tuple(row) for row in matrix)
    assert observation.board is not None


def test_scaled_capture_samples_logical_cells_across_the_full_image() -> None:
    matrix = [
        ["red", "yellow", "green", "blue"],
        ["yellow", "green", "blue", "red"],
        ["green", "blue", "red", "yellow"],
    ]
    logical = render_board(matrix, cell_size=10)
    scaled = logical.resize((logical.width * 2, logical.height * 2), Image.Resampling.NEAREST)
    spec = GridSpec(rows=3, cols=4)
    samples = sample_grid(scaled, Region(0, 0, logical.width, logical.height), spec)
    profile = calibrate(samples, num_colors=4)
    observation = classify_samples(samples, spec, profile, confidence_threshold=0.85)

    assert observation.valid
    assert observation.matrix == tuple(tuple(row) for row in matrix)
    # The diagnostic patch is in image pixels, not logical screen coordinates.
    assert samples[-1].patch_box == (65, 45, 74, 54)


def test_empty_cells_and_rightward_columns_are_not_a_color() -> None:
    matrix = [
        [None, None, None, None, None],
        [None, "blue", None, None, None],
        ["red", "blue", "yellow", None, None],
    ]
    image = render_board(matrix)
    spec = GridSpec(rows=3, cols=5)
    samples = sample_grid(image, Region(0, 0, image.width, image.height), spec)
    profile = calibrate(samples, num_colors=3)
    observation = classify_samples(samples, spec, profile, confidence_threshold=0.80)
    assert observation.valid
    assert observation.matrix[0] == (None, None, None, None, None)
    assert observation.board is not None
    assert observation.board.columns == (("red",), ("blue", "blue"), ("yellow",))


def test_small_rgb_perturbation_remains_confident() -> None:
    matrix = [["red", "yellow"], ["green", "blue"]]
    image = render_board(matrix)
    pixels = image.load()
    for x in range(image.width):
        for y in range(image.height):
            r, g, b = pixels[x, y]
            pixels[x, y] = (min(255, r + 3), max(0, g - 2), min(255, b + 2))
    spec = GridSpec(rows=2, cols=2)
    samples = sample_grid(image, Region(0, 0, image.width, image.height), spec)
    profile = calibrate(samples, num_colors=4)
    observation = classify_samples(samples, spec, profile, confidence_threshold=0.80)
    assert observation.valid
    assert observation.min_confidence >= 0.80


def test_unknown_color_is_rejected_instead_of_becoming_playable() -> None:
    spec = GridSpec(rows=1, cols=2)
    samples = (
        CellSample(0, 0, PALETTE["red"], (0, 0, 1, 1)),
        CellSample(0, 1, (210, 20, 210), (1, 0, 2, 1)),
    )
    profile = calibrate(
        (
            CellSample(0, 0, PALETTE["red"], (0, 0, 1, 1)),
            CellSample(0, 1, PALETTE["blue"], (1, 0, 2, 1)),
        ),
        num_colors=2,
    )
    observation = classify_samples(samples, spec, profile, confidence_threshold=0.8)
    assert not observation.valid
    assert observation.unknown_cells


def test_low_confidence_cell_blocks_automation() -> None:
    spec = GridSpec(rows=1, cols=2)
    profile = calibrate(
        (
            CellSample(0, 0, PALETTE["red"], (0, 0, 1, 1)),
            CellSample(0, 1, PALETTE["yellow"], (1, 0, 2, 1)),
        ),
        num_colors=2,
    )
    midpoint = tuple((red + yellow) // 2 for red, yellow in zip(PALETTE["red"], PALETTE["yellow"]))
    samples = (
        CellSample(0, 0, PALETTE["red"], (0, 0, 1, 1)),
        CellSample(0, 1, midpoint, (1, 0, 2, 1)),
    )
    observation = classify_samples(samples, spec, profile, confidence_threshold=0.99)
    assert not observation.valid
    assert observation.low_confidence_cells
