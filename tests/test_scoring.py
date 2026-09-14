import pytest

from blob_solver.game.groups import Move
from blob_solver.game.rules import GameRules, score_move


@pytest.mark.parametrize("size, expected", [(2, 4), (3, 9), (4, 16), (5, 25), (7, 49), (13, 169)])
def test_square_scoring(size: int, expected: int) -> None:
    move = Move.create("R", [(index, 0) for index in range(size)])
    assert GameRules().score_move(move) == expected
    assert score_move(move) == expected
    assert move.immediate_score == expected


def test_illegal_group_size_is_rejected() -> None:
    with pytest.raises(ValueError):
        GameRules().score_size(1)
