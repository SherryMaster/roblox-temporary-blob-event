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
def _find_groups_cached(board: Board, min_group: int) -> tuple[Move, ...]:
    visited: set[Cell] = set()
    groups: list[Move] = []
    for cell in board.cells():
        if cell in visited:
            continue
        color = board.cell_color(cell)
        stack = [cell]
        component: list[Cell] = []
        visited.add(cell)
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in _neighbors(current):
                if neighbor in visited:
                    continue
                try:
                    neighbor_color = board.cell_color(neighbor)
                except IndexError:
                    continue
                if neighbor_color == color:
                    visited.add(neighbor)
                    stack.append(neighbor)
        if len(component) >= min_group:
            groups.append(Move.create(color, component))
    groups.sort(key=lambda move: (move.cells[0], str(move.color)))
    return tuple(groups)


def find_groups(board: Board, min_group: int = 2) -> tuple[Move, ...]:
    """Return all orthogonally connected components of legal size."""

    if min_group < 2:
        raise ValueError("min_group must be at least 2")
    return _find_groups_cached(board, min_group)


def clear_group_cache() -> None:
    """Clear cached components, useful for long-running debug sessions/tests."""

    _find_groups_cached.cache_clear()
