"""Deterministic SameGame state transitions."""

from __future__ import annotations

from .board import Board, Cell
from .groups import Move, find_groups


class IllegalMoveError(ValueError):
    """Raised when a move is not exactly one legal connected component."""


def _validate_move(board: Board, move: Move, min_group: int) -> None:
    legal = find_groups(board, min_group=min_group)
    target = frozenset(move.cells)
    if any(candidate.color == move.color and frozenset(candidate.cells) == target for candidate in legal):
        return
    raise IllegalMoveError("move is not a complete legal component in this board")


def apply_move(
    board: Board,
    move: Move,
    *,
    min_group: int = 2,
    validate: bool = True,
) -> Board:
    """Remove a component, apply gravity, and compact columns left.

    Because columns are stored bottom-to-top, gravity is simply filtering the
    selected levels from each column. Removing empty columns then implements the
    leftward collapse while preserving the order of surviving columns.
    """

    if validate:
        _validate_move(board, move, min_group)
    removed: set[Cell] = set(move.cells)
    next_columns: list[tuple[object, ...]] = []
    for column_index, column in enumerate(board.columns):
        kept = tuple(
            value for row_from_bottom, value in enumerate(column)
            if (row_from_bottom, column_index) not in removed
        )
        if kept:
            next_columns.append(kept)
    return Board(next_columns)


def is_terminal(board: Board, *, min_group: int = 2) -> bool:
    return not find_groups(board, min_group=min_group)
