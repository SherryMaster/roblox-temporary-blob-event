from blob_solver.game.board import Board
from blob_solver.game.transition import apply_move
from threading import Event
from blob_solver.solver.beam import BeamConfig, BeamSolver
from blob_solver.solver.exact import ExactSolver
from blob_solver.solver.greedy import GreedySolver
from blob_solver.solver.rollout import RolloutConfig, RolloutSolver


def test_exact_solver_proves_simple_optimum() -> None:
    # Removing Y first makes the two red columns adjacent after compaction and
    # permits a single 4-cell red move: 16 instead of 4 + 4.
    board = Board.from_matrix(
        [
            "R.YR",
            "R.YR",
        ]
    )
    solution = ExactSolver().solve(board)
    assert solution.optimal_proven
    assert solution.total_score == 20
    assert solution.moves[0].color == "Y"


def test_exact_path_replays_to_its_reported_score() -> None:
    board = Board.from_matrix(["..G", "YYG", "RRG"])
    solution = ExactSolver().solve(board)
    current = board
    score = 0
    for move in solution.moves:
        score += move.immediate_score
        current = apply_move(current, move)
    assert score == solution.total_score
    assert current.block_count >= 0


def test_beam_returns_a_complete_best_known_solution() -> None:
    board = Board.from_matrix(["R.YR", "R.YR"])
    solution = BeamSolver(BeamConfig(beam_width=20, max_nodes=1000)).solve(board, time_limit=1)
    assert solution.total_score >= GreedySolver().solve(board).total_score
    assert solution.status == "BEST KNOWN SOLUTION"


def test_rollout_is_seeded_and_reproducible() -> None:
    board = Board.from_columns(
        [
            ("R", "R"),
            ("Y", "R", "R"),
            ("G", "G"),
            ("Y",),
        ]
    )
    config = RolloutConfig(seed=123, max_rollouts=50)
    first = RolloutSolver(config).solve(board, time_limit=None)
    second = RolloutSolver(config).solve(board, time_limit=None)
    assert (first.total_score, first.moves) == (second.total_score, second.moves)


def test_zero_budget_still_returns_fast_incumbent() -> None:
    board = Board.from_matrix(["RR", "YY"])
    solution = BeamSolver(BeamConfig(beam_width=10, max_nodes=100)).solve(board, time_limit=0)
    assert solution.moves
    assert solution.total_score >= 4
    assert not solution.optimal_proven


def test_cancelled_search_keeps_a_complete_incumbent() -> None:
    board = Board.from_matrix(["RR", "YY"])
    cancelled = Event()
    cancelled.set()
    solution = BeamSolver(BeamConfig(beam_width=10, max_nodes=100)).solve(
        board,
        time_limit=10,
        cancel_event=cancelled,
    )
    current = board
    for move in solution.moves:
        current = apply_move(current, move)
    assert current.block_count == 0
