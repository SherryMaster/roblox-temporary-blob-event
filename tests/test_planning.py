from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import random

from blob_solver.game.board import Board
from blob_solver.game.groups import find_groups
from blob_solver.game.transition import apply_move
from blob_solver.game.rules import GameRules
from blob_solver.solver.exact import ExactSolver
from blob_solver.solver.greedy import GreedySolver
from blob_solver.solver.hybrid import HybridConfig, HybridPlanner
from blob_solver.solver.plan import Plan


@lru_cache(maxsize=None)
def brute_force_score(board: Board) -> int:
    groups = find_groups(board)
    if not groups:
        return 0
    return max(
        move.size * move.size + brute_force_score(apply_move(board, move, validate=False))
        for move in groups
    )


def test_exact_solver_matches_brute_force_on_small_random_boards() -> None:
    rng = random.Random(20260914)
    checked = 0
    for width, height, samples in ((2, 2, 16), (3, 2, 16), (3, 3, 8)):
        for _ in range(samples):
            matrix = [[rng.choice(("R", "Y", "G", "B")) for _ in range(width)] for _ in range(height)]
            board = Board.from_matrix(matrix)
            result = ExactSolver().solve(board)
            assert result.optimal_proven
            assert result.total_score == brute_force_score(board)
            assert result.to_plan(board).replay_valid()
            checked += 1
    assert checked == 40


def test_exact_solver_publishes_memoized_value_and_principal_move() -> None:
    board = Board.from_matrix(["R.YR", "R.YR"])
    solver = ExactSolver()
    solution = solver.solve(board)

    assert solver.transposition_table is not None
    record = solver.transposition_table.exact(board)
    assert record is not None
    assert record.exact_future_score == solution.total_score
    assert record.best_move == solution.first_move


def test_exhaustive_hybrid_proof_is_replayable_with_a_tiny_frontier() -> None:
    board = Board.from_matrix(["R.YR", "R.YR"])
    solution = HybridPlanner(
        HybridConfig(
            quality="exhaustive",
            beam_width=1,
            candidate_moves=None,
            max_nodes=None,
            rollout_count=0,
        )
    ).solve(board)

    assert solution.optimal_proven
    assert solution.total_score == brute_force_score(board)
    assert solution.to_plan(board).replay_valid()


def test_plan_replay_preserves_custom_clear_board_scoring() -> None:
    board = Board.from_matrix(["RR"])
    rules = GameRules(clear_board_bonus=7)
    solution = ExactSolver(rules=rules).solve(board)
    plan = solution.to_plan(board, rules=rules)

    assert plan.total_score == 11
    assert plan.clear_board_bonus == 7
    assert plan.replay_valid()


def test_hybrid_plan_contains_every_replayed_intermediate_board() -> None:
    board = Board.from_matrix(["R.YR", "R.YR"])
    solution = HybridPlanner(
        HybridConfig(quality="fast", beam_width=40, max_nodes=2_000, candidate_moves=12)
    ).solve(board, time_limit=0.2)
    plan = solution.to_plan(board)

    assert plan.replay_valid()
    assert plan.total_score == sum(step.immediate_score for step in plan.steps)
    assert plan.steps[0].before_board == board
    assert not find_groups(plan.steps[-1].after_board)
    assert plan.total_score >= GreedySolver().solve(board).total_score


def test_hybrid_records_full_game_evidence_for_every_root_move() -> None:
    board = Board.from_matrix(["R.YR", "R.YR"])
    planner = HybridPlanner(HybridConfig(quality="fast", max_nodes=2_000))
    solution = planner.solve(board, time_limit=0.2)
    root_moves = find_groups(board)

    assert len(solution.root_evaluations) == len(root_moves)
    for evaluation in solution.root_evaluations:
        root_plan = Plan.from_moves(
            board,
            evaluation.best_complete_plan,
            upper_bound=evaluation.upper_bound,
        )
        assert root_plan.replay_valid()
        assert evaluation.best_complete_plan[0] == evaluation.first_move
        assert evaluation.best_complete_score_found == sum(
            move.size * move.size for move in evaluation.best_complete_plan
        )
        assert evaluation.upper_bound >= evaluation.best_complete_score_found


def test_hybrid_cancellation_still_returns_a_complete_plan() -> None:
    from threading import Event

    board = Board.from_matrix(["R.YR", "R.YR"])
    cancelled = Event()
    cancelled.set()
    solution = HybridPlanner(HybridConfig(quality="balanced")).solve(
        board,
        time_limit=10,
        cancel_event=cancelled,
    )
    assert solution.cancelled
    assert solution.to_plan(board).replay_valid()


def test_larger_deterministic_node_budget_keeps_the_same_incumbent_or_improves_it() -> None:
    board = Board.from_matrix(["R.YR", "R.YR"])
    scores = []
    for node_budget in (1, 100):
        solution = HybridPlanner(
            HybridConfig(
                quality="fast",
                beam_width=40,
                max_nodes=node_budget,
                candidate_moves=12,
            )
        ).solve(board, time_limit=0.2)
        scores.append(solution.total_score)
    assert scores[1] >= scores[0]


def test_column_collapse_can_beat_the_largest_immediate_group() -> None:
    board = Board.from_matrix(
        [
            ["R", None, "R"],
            ["R", None, "R"],
            ["R", "Y", "R"],
            ["R", "Y", "R"],
        ]
    )
    greedy = GreedySolver().solve(board)
    exact = ExactSolver().solve(board)

    assert greedy.total_score == 36
    assert exact.total_score == 68
    assert exact.moves[0].color == "Y"
    assert exact.moves[0].size == 2
    assert exact.to_plan(board).replay_valid()


def test_gravity_can_turn_a_small_blocker_removal_into_a_large_group() -> None:
    board = Board.from_matrix(
        [
            ["G", "Y", "G"],
            ["G", "Y", "Y"],
            ["Y", "R", "R"],
            ["G", "Y", "G"],
        ]
    )
    greedy = GreedySolver().solve(board)
    exact = ExactSolver().solve(board)
    after_first = apply_move(board, exact.moves[0])

    assert exact.moves[0].color == "R"
    assert exact.moves[0].size == 2
    assert max(move.size for move in find_groups(board)) == 3
    assert max(move.size for move in find_groups(after_first)) == 5
    assert greedy.total_score == 21
    assert exact.total_score == 54
    assert exact.to_plan(board).replay_valid()


def test_reference_generation_greedy_regression_and_hybrid_quality() -> None:
    fixture = Path(__file__).parents[1] / "examples" / "real_generation_001.txt"
    matrix = [line.split() for line in fixture.read_text(encoding="utf-8").splitlines() if line and not line.startswith("#")]
    board = Board.from_matrix(matrix)

    greedy_score = GreedySolver().solve(board).total_score
    # Use the normal Fast wall-clock budget, but keep this regression serial so
    # process-pool startup variance cannot decide whether the quality floor is
    # exercised. The separate anytime tests cover zero/tiny budgets.
    fast_plan = HybridPlanner(
        HybridConfig(quality="fast", process_count=1)
    ).solve(board, time_limit=1.0).to_plan(board)

    assert board.block_count == 100
    assert greedy_score == 288
    assert fast_plan.replay_valid()
    assert fast_plan.total_score >= 436
