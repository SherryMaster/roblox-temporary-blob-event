from __future__ import annotations

from blob_solver.game.board import Board
from blob_solver.vision.classifier import BoardObservation
from blob_solver.vision.region import Region
from blob_solver.vision.settle import BoardSettler


def _observation(board: Board) -> BoardObservation:
    matrix = board.to_matrix(empty=None)
    return BoardObservation(
        rows=len(matrix),
        cols=len(matrix[0]) if matrix else 0,
        cells=(),
        matrix=matrix,
        board=board,
        confidence_threshold=0.9,
    )


def test_region_parses_explicit_and_slurp_forms() -> None:
    assert Region.parse("100,200,820,740") == Region(100, 200, 820, 740)
    assert Region.parse("100,200 820x740") == Region(100, 200, 820, 740)
    assert Region.parse("100,200 820 740") == Region(100, 200, 820, 740)
    assert Region(10, 20, 100, 100).cell_center(2, 2, 1, 0) == (35, 95)


def test_settle_requires_repeated_identical_valid_boards() -> None:
    first = Board.from_columns([("R", "R"), ("G",)])
    second = Board.from_columns([("R",), ("G", "G")])
    observations = iter([_observation(first), _observation(second), _observation(second), _observation(second)])
    result = BoardSettler(settle_frames=3, interval_seconds=0.001, timeout_seconds=0.2).wait_for_settled(
        lambda: next(observations)
    )
    assert result.board == second
    assert result.frames == 4
