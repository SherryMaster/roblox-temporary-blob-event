"""Connected-component detection for orthogonally adjacent cells."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

from .board import Board, Cell, Color


@dataclass(frozen=True, slots=True)
class Move:
    """A complete legal component in one board state.

    Coordinates are ``(row_from_bottom, column)``. They are intentionally
    logical coordinates; the desktop layer resolves them to pixels later.
    """

    color: Color
    cells: tuple[Cell, ...]
    immediate_score: int

    @classmethod
    def create(cls, color: Color, cells: Iterable[Cell]) -> "Move":
        ordered = tuple(sorted(set(cells)))
        if not ordered:
            raise ValueError("a move must contain at least one cell")
        return cls(color=color, cells=ordered, immediate_score=len(ordered) ** 2)

    @property
    def size(self) -> int:
        return len(self.cells)

    @property
    def click_cell(self) -> Cell:
        """Choose a central, deterministic member for a physical click."""

        return self.cells[len(self.cells) // 2]


def _neighbors(cell: Cell) -> tuple[Cell, ...]:
    row, column = cell
    return ((row + 1, column), (row - 1, column), (row, column - 1), (row, column + 1))


@lru_cache(maxsize=100_000)
def _find_components_cached(board: Board) -> tuple[Move, ...]:
    columns = board.columns
    visited: set[Cell] = set()
    components: list[Move] = []
    for column_index, column in enumerate(columns):
        for row_from_bottom in range(len(column)):
            cell = (row_from_bottom, column_index)
            if cell in visited:
                continue
            color = column[row_from_bottom]
            stack = [cell]
            component: list[Cell] = []
            visited.add(cell)
            while stack:
                current_row, current_column = stack.pop()
                current = (current_row, current_column)
                component.append(current)
                for neighbor_row, neighbor_column in _neighbors(current):
                    if neighbor_row < 0 or neighbor_column < 0:
                        continue
                    if neighbor_column >= len(columns) or neighbor_row >= len(columns[neighbor_column]):
                        continue
                    neighbor = (neighbor_row, neighbor_column)
                    if neighbor in visited or columns[neighbor_column][neighbor_row] != color:
                        continue
                    visited.add(neighbor)
                    stack.append(neighbor)
            components.append(Move.create(color, component))
    components.sort(key=lambda move: (move.cells[0], str(move.color)))
    return tuple(components)


def find_components(board: Board) -> tuple[Move, ...]:
    """Return every orthogonal component, including singleton cells.

    Search heuristics use singleton topology even though singletons are not
    playable moves. Keeping this separate from ``find_groups`` preserves the
    public legal-move contract and lets the component traversal stay cached.
    """

    return _find_components_cached(board)


def _find_groups_cached(board: Board, min_group: int) -> tuple[Move, ...]:
    return tuple(component for component in find_components(board) if component.size >= min_group)


def find_groups(board: Board, min_group: int = 2) -> tuple[Move, ...]:
    """Return all orthogonally connected components of legal size."""

    if min_group < 2:
        raise ValueError("min_group must be at least 2")
    return _find_groups_cached(board, min_group)


def clear_group_cache() -> None:
    """Clear cached components, useful for long-running debug sessions/tests."""

    _find_components_cached.cache_clear()
