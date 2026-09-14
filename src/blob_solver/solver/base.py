"""Shared solver data structures and cancellation/time-budget support."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Event
from time import monotonic
from typing import Protocol

from blob_solver.game.board import Board
from blob_solver.game.groups import Move
from blob_solver.game.rules import GameRules


@dataclass(slots=True)
class SearchStats:
    nodes: int = 0
    started_at: float = field(default_factory=monotonic)
    finished_at: float | None = None
    interrupted: bool = False
    optimality_proven: bool = False

    @property
    def elapsed(self) -> float:
        return (self.finished_at or monotonic()) - self.started_at


@dataclass(frozen=True, slots=True)
class Solution:
    """A plan of logical moves for one observed board."""

    moves: tuple[Move, ...]
    total_score: int
    optimal_proven: bool
    search_time_seconds: float
    nodes_examined: int
    solver_name: str
    upper_bound: int | None = None
    cancelled: bool = False

    @property
    def first_move(self) -> Move | None:
        return self.moves[0] if self.moves else None

    @property
    def move_count(self) -> int:
        return len(self.moves)

    @property
    def status(self) -> str:
        return "OPTIMAL SOLUTION PROVEN" if self.optimal_proven else "BEST KNOWN SOLUTION"


class Solver(Protocol):
    def solve(
        self,
        board: Board,
        *,
        time_limit: float | None = None,
        cancel_event: Event | None = None,
    ) -> Solution:
        ...


def score_transition(board: Board, move: Move, rules: GameRules) -> int:
    """Score a move through GameRules, including future clear-board variants."""

    return rules.score_move(move, cleared_board=board.block_count == move.size)


class SearchBudget:
    """A cooperative wall-clock/node budget shared by recursive searches."""

    __slots__ = ("deadline", "cancel_event", "max_nodes", "stats")

    def __init__(
        self,
        *,
        time_limit: float | None,
        cancel_event: Event | None,
        max_nodes: int | None,
        stats: SearchStats,
    ) -> None:
        self.deadline = None if time_limit is None else monotonic() + max(0.0, time_limit)
        self.cancel_event = cancel_event
        self.max_nodes = max_nodes
        self.stats = stats

    def expired(self) -> bool:
        if self.cancel_event is not None and self.cancel_event.is_set():
            return True
        if self.deadline is not None and monotonic() >= self.deadline:
            return True
        if self.max_nodes is not None and self.stats.nodes >= self.max_nodes:
            return True
        return False

    def visit(self) -> bool:
        if self.expired():
            self.stats.interrupted = True
            return False
        self.stats.nodes += 1
        return True


def finish_solution(
    *,
    moves: list[Move] | tuple[Move, ...],
    score: int,
    stats: SearchStats,
    solver_name: str,
    optimal_proven: bool,
    upper_bound: int | None = None,
) -> Solution:
    stats.finished_at = monotonic()
    stats.optimality_proven = optimal_proven
    return Solution(
        moves=tuple(moves),
        total_score=score,
        optimal_proven=optimal_proven,
        search_time_seconds=stats.elapsed,
        nodes_examined=stats.nodes,
        solver_name=solver_name,
        upper_bound=upper_bound,
        cancelled=stats.interrupted,
    )
