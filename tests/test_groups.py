from blob_solver.game.board import Board
from blob_solver.game.groups import find_groups


def test_horizontal_group() -> None:
    board = Board.from_matrix(["...", "RR."])
    groups = find_groups(board)
    assert len(groups) == 1
    assert groups[0].color == "R"
    assert set(groups[0].cells) == {(0, 0), (0, 1)}


def test_vertical_group() -> None:
    board = Board.from_matrix(["R.", "R.", "R."])
    assert find_groups(board)[0].size == 3
    assert set(find_groups(board)[0].cells) == {(0, 0), (1, 0), (2, 0)}


def test_l_shape_group() -> None:
    board = Board.from_matrix(["R..", "R..", "RR."])
    assert find_groups(board)[0].size == 4


def test_diagonal_touch_is_disconnected() -> None:
    board = Board.from_matrix(["R.", "BR"])
    assert find_groups(board) == ()


def test_multiple_components_and_singletons() -> None:
    board = Board.from_columns(
        [
            ("R", "R"),
            ("G",),
            ("R", "R"),
            ("B", "B", "B"),
            ("Y",),
        ]
    )
    groups = find_groups(board)
    assert sorted((group.color, group.size) for group in groups) == [
        ("B", 3),
        ("R", 2),
        ("R", 2),
    ]
