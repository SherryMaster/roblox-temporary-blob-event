"""Turn calibrated cell samples into a validated board observation."""

from __future__ import annotations

from dataclasses import dataclass

from blob_solver.game.board import Board, BoardInvariantError

from .calibration import CalibrationProfile, calibrate as calibrate_profile
from .grid import CellSample, GridSpec


@dataclass(frozen=True, slots=True)
class CellClassification:
    row: int
    col: int
    symbol: str | None
    confidence: float
    rgb: tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class BoardObservation:
    """A fixed-grid visual result, including uncertainty rather than hiding it."""

    rows: int
    cols: int
    cells: tuple[CellClassification, ...]
    matrix: tuple[tuple[str | None, ...], ...]
    board: Board | None
    confidence_threshold: float
    error: str | None = None

    @property
    def low_confidence_cells(self) -> tuple[CellClassification, ...]:
        return tuple(cell for cell in self.cells if cell.confidence < self.confidence_threshold)

    @property
    def unknown_cells(self) -> tuple[CellClassification, ...]:
        return tuple(cell for cell in self.cells if cell.symbol == "?")

    @property
    def valid(self) -> bool:
        return self.board is not None and not self.low_confidence_cells and not self.unknown_cells and self.error is None

    @property
    def min_confidence(self) -> float:
        return min((cell.confidence for cell in self.cells), default=0.0)

    def text(self) -> str:
        def label(cell: str | None) -> str:
            return "." if cell is None else cell

        return "\n".join(" ".join(label(value) for value in row) for row in self.matrix)


def classify_samples(
    samples: tuple[CellSample, ...],
    spec: GridSpec,
    profile: CalibrationProfile,
    *,
    confidence_threshold: float = 0.90,
) -> BoardObservation:
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be between 0 and 1")
    by_coordinate = {(sample.row, sample.col): sample for sample in samples}
    classifications: list[CellClassification] = []
    matrix: list[list[str | None]] = [[None for _ in range(spec.cols)] for _ in range(spec.rows)]
    for row in range(spec.rows):
        for col in range(spec.cols):
            sample = by_coordinate.get((row, col))
            if sample is None:
                raise ValueError(f"missing sample for cell ({row}, {col})")
            symbol, confidence = profile.classify(sample.rgb)
            cell = CellClassification(row, col, symbol, confidence, sample.rgb)
            classifications.append(cell)
            matrix[row][col] = symbol
    error: str | None = None
    board: Board | None = None
    try:
        if any(cell.symbol == "?" for cell in classifications):
            error = "unknown color classification"
        elif any(cell.confidence < confidence_threshold for cell in classifications):
            error = "one or more cells are below the confidence threshold"
        else:
            board = Board.from_matrix(matrix, empty=None, strict=True)
    except BoardInvariantError as exc:
        error = f"invalid settled board geometry: {exc}"
    return BoardObservation(
        rows=spec.rows,
        cols=spec.cols,
        cells=tuple(classifications),
        matrix=tuple(tuple(row) for row in matrix),
        board=board,
        confidence_threshold=confidence_threshold,
        error=error,
    )


class VisionClassifier:
    """Stateful convenience wrapper for calibration followed by repeated scans."""

    def __init__(
        self,
        profile: CalibrationProfile | None = None,
        *,
        confidence_threshold: float = 0.90,
    ) -> None:
        self.profile = profile
        self.confidence_threshold = confidence_threshold

    def calibrate(
        self,
        samples: tuple[CellSample, ...],
        *,
        num_colors: int = 4,
        symbols: tuple[str, ...] = ("red", "yellow", "green", "blue"),
    ) -> CalibrationProfile:
        self.profile = calibrate_profile(samples, num_colors=num_colors, symbols=symbols)
        return self.profile

    def classify(
        self,
        samples: tuple[CellSample, ...],
        spec: GridSpec,
    ) -> BoardObservation:
        if self.profile is None:
            raise ValueError("classifier has not been calibrated")
        return classify_samples(
            samples,
            spec,
            self.profile,
            confidence_threshold=self.confidence_threshold,
        )
