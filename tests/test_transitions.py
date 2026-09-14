import pytest

from blob_solver.game.board import Board, BoardInvariantError
from blob_solver.game.groups import Move, find_groups
from blob_solver.game.transition import IllegalMoveError, apply_move, is_terminal


def test_known_transition_includes_gravity_and_column_shift() -> None:
    board = Board.from_columns(
        [
            ("Y", "R", "R"),
            ("B", "B"),
            ("G", "G", "G"),
        ]
    )
    move = next(group for group in find_groups(board) if group.color == "B")
    result = apply_move(board, move)
    assert result.to_matrix(rows=3, cols=4, empty=".") == (
        ("R", "G", ".", "."),
        ("R", "G", ".", "."),
        ("Y", "G", ".", "."),
    )


def test_partial_component_is_not_accepted() -> None:
    board = Board.from_matrix(["RR"])
    partial = Move.create("R", [(0, 0)])
    with pytest.raises(IllegalMoveError):
        apply_move(board, partial)


def test_terminal_detection() -> None:
    assert is_terminal(Board.from_matrix(["...", "R.Y"]))
    assert not is_terminal(Board.from_matrix(["RR"]))


def test_internal_holes_are_rejected() -> None:
    with pytest.raises(BoardInvariantError):
        Board.from_matrix(["R", ".", "R"])
