"""Immutable canonical board representation.

The solver intentionally does not store a rectangular array. A settled board is
just its surviving columns, and every column is stored bottom-to-top. This makes
gravity a filter operation and makes equivalent visual states hash identically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Iterable, Sequence, TypeAlias

Color: TypeAlias = Hashable
Cell: TypeAlias = tuple[int, int]


class BoardInvariantError(ValueError):
    """Raised when a matrix or compact board violates settled-board rules."""


def _is_empty(value: object, empty: object | None) -> bool:
    return value is None or value == "." or (empty is not None and value == empty)


@dataclass(frozen=True, slots=True, init=False)
class Board:
    """Canonical SameGame state.

    ``columns[c][r]`` is the color at column ``c`` and row ``r`` counted from
    the bottom. Empty columns are not stored. The constructor normalizes nested
    sequences to tuples and rejects empty columns.
    """

    columns: tuple[tuple[Color, ...], ...]

    def __init__(self, columns: Iterable[Iterable[Color]] = ()) -> None:
        normalized = tuple(tuple(column) for column in columns)
        if any(not column for column in normalized):
            raise BoardInvariantError("canonical boards cannot contain empty columns")
        for column_index, column in enumerate(normalized):
            for value in column:
                if value is None or value == ".":
                    raise BoardInvariantError(
                        f"empty value in compact column {column_index}; use matrix conversion"
                    )
        object.__setattr__(self, "columns", normalized)

    @classmethod
    def from_columns(cls, columns: Iterable[Iterable[Color]]) -> "Board":
        """Build a canonical board from bottom-to-top columns."""

        return cls(columns)

    @classmethod
    def from_matrix(
        cls,
        matrix: Sequence[Sequence[Color | None] | str],
        *,
        empty: object | None = None,
        strict: bool = True,
    ) -> "Board":
        """Convert a top-to-bottom matrix to compact columns.

        Empty cells may occur above occupied cells. If ``strict`` is true, an
        occupied cell below an empty cell within a column is rejected because it
        represents a hole that gravity should already have removed.
        """

        rows = [tuple(row) if isinstance(row, str) else tuple(row) for row in matrix]
        if not rows:
            return cls()
        width = len(rows[0])
        if width == 0:
            return cls()
        if any(len(row) != width for row in rows):
            raise BoardInvariantError("matrix rows must all have the same width")

        columns: list[tuple[Color, ...]] = []
        for column_index in range(width):
            bottom_to_top: list[Color] = []
            saw_empty = False
            for row_index in range(len(rows) - 1, -1, -1):
                value = rows[row_index][column_index]
                if _is_empty(value, empty):
                    saw_empty = True
                    continue
                if strict and saw_empty:
                    raise BoardInvariantError(
                        "matrix contains a hole in column "
                        f"{column_index} at row {row_index}"
                    )
                bottom_to_top.append(value)  # type: ignore[arg-type]
            if bottom_to_top:
                columns.append(tuple(bottom_to_top))
        return cls(columns)

    def to_matrix(
        self,
        *,
        rows: int | None = None,
        cols: int | None = None,
        empty: Color | None = None,
    ) -> tuple[tuple[Color | None, ...], ...]:
        """Return a top-to-bottom rectangular matrix.

        ``rows`` and ``cols`` can restore the configured screen grid. They may
        be larger than the compact board, but never smaller.
        """

        height = max((len(column) for column in self.columns), default=0)
        width = len(self.columns)
        output_rows = height if rows is None else rows
        output_cols = width if cols is None else cols
        if output_rows < height:
            raise ValueError(f"requested {output_rows} rows, board needs {height}")
        if output_cols < width:
            raise ValueError(f"requested {output_cols} columns, board needs {width}")
        matrix: list[list[Color | None]] = [
            [empty for _ in range(output_cols)] for _ in range(output_rows)
        ]
        for column_index, column in enumerate(self.columns):
            for row_from_bottom, value in enumerate(column):
                matrix[output_rows - 1 - row_from_bottom][column_index] = value
        return tuple(tuple(row) for row in matrix)

    @property
    def width(self) -> int:
        """Number of non-empty columns in the compact state."""

        return len(self.columns)

    @property
    def height(self) -> int:
        """Height of the tallest surviving column."""

        return max((len(column) for column in self.columns), default=0)

    @property
    def block_count(self) -> int:
        return sum(len(column) for column in self.columns)

    def color_counts(self) -> dict[Color, int]:
        counts: dict[Color, int] = {}
        for column in self.columns:
            for color in column:
                counts[color] = counts.get(color, 0) + 1
        return counts

    def cell_color(self, cell: Cell) -> Color:
        """Return a cell color using ``(row_from_bottom, column)`` coordinates."""

        row_from_bottom, column = cell
        if column < 0 or column >= self.width:
            raise IndexError(f"column out of range: {column}")
        if row_from_bottom < 0 or row_from_bottom >= len(self.columns[column]):
            raise IndexError(f"row out of range: {row_from_bottom}")
        return self.columns[column][row_from_bottom]

    def cells(self) -> tuple[Cell, ...]:
        """Return every occupied logical cell in deterministic order."""

        return tuple(
            (row_from_bottom, column_index)
            for column_index, column in enumerate(self.columns)
            for row_from_bottom in range(len(column))
        )

    def __str__(self) -> str:
        matrix = self.to_matrix(empty=".")
        return "\n".join(" ".join(str(value) for value in row) for row in matrix)
