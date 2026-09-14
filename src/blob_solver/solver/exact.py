"""Exact depth-first branch-and-bound solver for small/deep searches."""

from __future__ import annotations

from threading import Event

from blob_solver.game.board import Board
from blob_solver.game.groups import Move, find_groups
from blob_solver.game.rules import GameRules
from blob_solver.game.transition import apply_move

from .base import SearchBudget, SearchStats, Solution, finish_solution, score_transition
from .greedy import GreedySolver
from .heuristics import color_ideal_upper_bound, ordered_moves
from .transposition import TranspositionTable


class ExactSolver:
    """Prove the optimum when the search finishes within its budget.

    The color-count square sum is an admissible upper bound: no future sequence
    can score more than if all remaining cells of each color could be merged into
    one component. A transposition record only prunes a weaker path reaching the
    same state; it never changes the exact result of a stronger path.
    """

    def __init__(
        self,
        *,
        max_nodes: int | None = None,
        min_group: int = 2,
        rules: GameRules | None = None,
    ) -> None:
        self.max_nodes = max_nodes
        if min_group < 2:
            raise ValueError("min_group must be at least 2")
        self.min_group = min_group
        self.rules = rules or GameRules(min_group=min_group)

    def solve(
        self,
        board: Board,
        *,
        time_limit: float | None = None,
        cancel_event: Event | None = None,
    ) -> Solution:
        stats = SearchStats()
        budget = SearchBudget(
            time_limit=time_limit,
            cancel_event=cancel_event,
            max_nodes=self.max_nodes,
            stats=stats,
        )
        incumbent = GreedySolver(
            min_group=self.min_group,
            rules=self.rules,
        ).solve(board)
        best_score = incumbent.total_score
        best_moves = list(incumbent.moves)
        table = TranspositionTable(max_entries=None)

        def dfs(current: Board, score_so_far: int, path: list[Move]) -> None:
            nonlocal best_score, best_moves
            if not budget.visit():
                return
            optimistic = score_so_far + color_ideal_upper_bound(current, self.rules)
            if optimistic <= best_score:
                return
            previous = table.get(current)
            if previous is not None and previous.best_score_reaching >= score_so_far:
                return
            table.update_reaching(current, score_so_far)
            legal = ordered_moves(current, min_group=self.min_group)
            if not legal:
                if score_so_far > best_score:
                    best_score = score_so_far
                    best_moves = list(path)
                return
            fully_explored = True
            for move in legal:
                if budget.expired():
                    fully_explored = False
                    break
                next_board = apply_move(current, move, validate=False)
                path.append(move)
                dfs(next_board, score_so_far + score_transition(current, move, self.rules), path)
                path.pop()
            # This record means only that this reaching path was explored far
            # enough for pruning; exact future values are not needed by the
            # branch-and-bound proof and are intentionally not fabricated.
            if not fully_explored:
                stats.interrupted = True

        dfs(board, 0, [])
        interrupted = stats.interrupted or budget.expired()
        # A complete root traversal, including branches proven impossible by the
        # admissible bound, is a mathematical proof. A timeout/cancel is not.
        proven = not interrupted
        return finish_solution(
            moves=best_moves,
            score=best_score,
            stats=stats,
            solver_name="exact-branch-and-bound",
            optimal_proven=proven,
            upper_bound=best_score if proven else color_ideal_upper_bound(board, self.rules),
        )
