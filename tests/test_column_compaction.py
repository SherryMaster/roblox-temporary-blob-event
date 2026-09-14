from blob_solver.game.board import Board
from blob_solver.game.groups import find_groups
from blob_solver.game.transition import apply_move


def test_empty_columns_are_removed_and_order_is_preserved() -> None:
    board = Board.from_columns(
        [
            ("A", "A"),
            ("B", "B"),
            ("C", "C"),
            ("D",),
            ("E", "E"),
        ]
    )
    move = next(group for group in find_groups(board) if group.color == "B")
    assert apply_move(board, move).columns == (("A", "A"), ("C", "C"), ("D",), ("E", "E"))


def test_board_from_matrix_compacts_empty_columns() -> None:
    board = Board.from_matrix(["A..C", "A..C"])
    assert board.columns == (("A", "A"), ("C", "C"))
