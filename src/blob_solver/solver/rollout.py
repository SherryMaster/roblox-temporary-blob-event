"""Randomized greedy rollouts / random restarts for anytime discovery."""

from __future__ import annotations

import random
from dataclasses import dataclass
from threading import Event
from time import monotonic

from blob_solver.game.board import Board
from blob_solver.game.groups import Move, find_groups
from blob_solver.game.rules import GameRules
from blob_solver.game.transition import apply_move

from .base import SearchStats, Solution, finish_solution, score_transition
from .greedy import GreedySolver
from .heuristics import color_ideal_upper_bound, evaluate_board, ordered_moves


@dataclass(frozen=True, slots=True)
class RolloutConfig:
    seed: int = 0
    candidate_moves: int = 8
    exploration: float = 0.35
    max_rollouts: int | None = None
    min_group: int = 2

    def __post_init__(self) -> None:
        if self.candidate_moves < 1:
            raise ValueError("candidate_moves must be positive")
        if self.exploration < 0:
            raise ValueError("exploration must not be negative")
        if self.min_group < 2:
            raise ValueError("min_group must be at least 2")


class RolloutSolver:
    def __init__(self, config: RolloutConfig | None = None, *, rules: GameRules | None = None) -> None:
        self.config = config or RolloutConfig()
        self.rules = rules or GameRules(min_group=self.config.min_group)

    def solve(
        self,
        board: Board,
        *,
        time_limit: float | None = None,
        cancel_event: Event | None = None,
    ) -> Solution:
        stats = SearchStats()
        started = monotonic()
        deadline = None if time_limit is None else started + max(0.0, time_limit)
        rng = random.Random(self.config.seed)
        incumbent = GreedySolver(
            min_group=self.config.min_group,
            rules=self.rules,
        ).solve(board)
        best_moves = list(incumbent.moves)
        best_score = incumbent.total_score
        rollouts = 0
        max_rollouts = self.config.max_rollouts
        if max_rollouts is None and time_limit is None:
            max_rollouts = 1000

        while max_rollouts is None or rollouts < max_rollouts:
            if (deadline is not None and monotonic() >= deadline) or (
                cancel_event is not None and cancel_event.is_set()
            ):
                stats.interrupted = True
                break
            rollouts += 1
            current = board
            path: list[Move] = []
            score = 0
            while True:
                if deadline is not None and monotonic() >= deadline:
                    stats.interrupted = True
                    break
                if cancel_event is not None and cancel_event.is_set():
                    stats.interrupted = True
                    break
                legal = ordered_moves(current, min_group=self.config.min_group)
                stats.nodes += 1
                if not legal:
                    if score > best_score:
                        best_score, best_moves = score, list(path)
                    break
                candidates = legal[: self.config.candidate_moves]
                weights = [
                    max(
                        0.001,
                        move.immediate_score
                        + self.config.exploration
                        * evaluate_board(
                            apply_move(current, move, validate=False),
                            min_group=self.config.min_group,
                        ),
                    )
                    for move in candidates
                ]
                move = rng.choices(candidates, weights=weights, k=1)[0]
                path.append(move)
                score += score_transition(current, move, self.rules)
                current = apply_move(current, move, validate=False)
            if stats.interrupted:
                break
        return finish_solution(
            moves=best_moves,
            score=best_score,
            stats=stats,
            solver_name="randomized-rollout",
            optimal_proven=False,
            upper_bound=color_ideal_upper_bound(board, self.rules),
        )
