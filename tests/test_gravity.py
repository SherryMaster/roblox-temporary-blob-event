from blob_solver.game.board import Board
from blob_solver.game.groups import find_groups
from blob_solver.game.transition import apply_move


def test_gravity_filters_each_column_independently() -> None:
    board = Board.from_matrix(
        [
            "UV",
            "RR",
            "YB",
        ]
    )
    move = next(group for group in find_groups(board) if group.color == "R")
    result = apply_move(board, move)
    assert result.columns == (("Y", "U"), ("B", "V"))


def test_deleting_bottom_cell_makes_upper_cells_fall() -> None:
    board = Board.from_matrix(["R", "B", "B"])
    move = find_groups(board)[0]
    assert apply_move(board, move).columns == (("R",),)
