"""Greedy baselines used for comparison and as a fast incumbent."""

from __future__ import annotations

from threading import Event

from blob_solver.game.board import Board
from blob_solver.game.groups import Move, find_groups
from blob_solver.game.rules import GameRules
from blob_solver.game.transition import apply_move

from .base import SearchStats, Solution, finish_solution, score_transition
from .heuristics import color_ideal_upper_bound


class GreedySolver:
    def __init__(
        self,
        *,
        policy: str = "score",
        min_group: int = 2,
        rules: GameRules | None = None,
    ) -> None:
        if policy not in {"score", "size"}:
            raise ValueError("greedy policy must be 'score' or 'size'")
        self.policy = policy
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
        current = board
        moves: list[Move] = []
        score = 0
        while True:
            if cancel_event is not None and cancel_event.is_set():
                stats.interrupted = True
                break
            legal = find_groups(current, min_group=self.min_group)
            if not legal:
                break
            stats.nodes += 1
            if self.policy == "score":
                move = max(
                    legal,
                    key=lambda candidate: (
                        score_transition(current, candidate, self.rules),
                        candidate.size,
                        candidate.cells,
                    ),
                )
            else:
                move = max(
                    legal,
                    key=lambda candidate: (
                        candidate.size,
                        score_transition(current, candidate, self.rules),
                        candidate.cells,
                    ),
                )
            moves.append(move)
            score += score_transition(current, move, self.rules)
            current = apply_move(current, move, min_group=self.min_group, validate=False)
        return finish_solution(
            moves=moves,
            score=score,
            stats=stats,
            solver_name=f"greedy-{self.policy}",
            # A greedy completion is never a proof of global optimality. The
            # exact solver owns that label, even when this happens to be optimal.
            optimal_proven=False,
            upper_bound=color_ideal_upper_bound(board, self.rules),
        )
