"""Replay-valid complete plans for one deterministic SameGame generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from blob_solver.game.board import Board, Cell
from blob_solver.game.groups import Move, find_groups
from blob_solver.game.rules import GameRules
from blob_solver.game.transition import apply_move, is_terminal


class PlanInvariantError(ValueError):
    """Raised when a proposed plan is not a legal complete replay."""


@dataclass(frozen=True, slots=True)
class PlanStep:
    """One frozen, simulated move in a complete plan.

    ``index`` is one-based for display. Both board states are retained so the
    UI can inspect future states without touching the screen or re-solving.
    """

    index: int
    before_board: Board
    move: Move
    after_board: Board
    group_size: int
    immediate_score: int
    cumulative_score: int
    click_cell: Cell


@dataclass(frozen=True, slots=True)
class RootMoveEvaluation:
    """Full-game evidence collected for one legal root move."""

    first_move: Move
    best_complete_score_found: int
    best_complete_plan: tuple[Move, ...]
    upper_bound: int
    search_nodes: int


@dataclass(frozen=True, slots=True)
class Plan:
    """An immutable solution for a single observed initial board."""

    initial_board: Board
    steps: tuple[PlanStep, ...]
    total_score: int
    solved_score: int
    upper_bound: int
    optimal_proven: bool
    solver_stats: Mapping[str, object]
    root_evaluations: tuple[RootMoveEvaluation, ...] = ()
    min_group: int = 2
    clear_board_bonus: int = 0

    def __post_init__(self) -> None:
        if self.total_score < 0 or self.solved_score < 0:
            raise PlanInvariantError("plan scores cannot be negative")
        if self.solved_score != self.total_score:
            raise PlanInvariantError("solved_score must equal the replayed plan score")
        if self.upper_bound < self.total_score:
            raise PlanInvariantError("upper bound cannot be below the complete score")
        if self.min_group < 2:
            raise PlanInvariantError("min_group must be at least two")
        if self.clear_board_bonus < 0:
            raise PlanInvariantError("clear_board_bonus cannot be negative")
        if not self.replay_valid():
            raise PlanInvariantError("plan failed replay validation")

    @classmethod
    def from_moves(
        cls,
        initial_board: Board,
        moves: tuple[Move, ...] | list[Move],
        *,
        rules: GameRules | None = None,
        upper_bound: int | None = None,
        optimal_proven: bool = False,
        solver_stats: Mapping[str, object] | None = None,
        root_evaluations: tuple[RootMoveEvaluation, ...] = (),
    ) -> "Plan":
        """Simulate and validate every step before publishing a plan."""

        active_rules = rules or GameRules()
        current = initial_board
        cumulative = 0
        steps: list[PlanStep] = []
        for index, requested_move in enumerate(moves, start=1):
            legal = find_groups(current, min_group=active_rules.min_group)
            move = next(
                (
                    candidate
                    for candidate in legal
                    if candidate.color == requested_move.color
                    and frozenset(candidate.cells) == frozenset(requested_move.cells)
                ),
                None,
            )
            if move is None:
                raise PlanInvariantError(
                    f"step {index} is not a complete legal component for its before_board"
                )
            after = apply_move(current, move, min_group=active_rules.min_group, validate=False)
            immediate = active_rules.score_move(move, cleared_board=after.block_count == 0)
            cumulative += immediate
            steps.append(
                PlanStep(
                    index=index,
                    before_board=current,
                    move=move,
                    after_board=after,
                    group_size=move.size,
                    immediate_score=immediate,
                    cumulative_score=cumulative,
                    click_cell=move.click_cell,
                )
            )
            current = after

        if not is_terminal(current, min_group=active_rules.min_group):
            raise PlanInvariantError("plan is partial; legal moves remain after its last step")
        bound = cumulative if upper_bound is None else max(cumulative, int(upper_bound))
        return cls(
            initial_board=initial_board,
            steps=tuple(steps),
            total_score=cumulative,
            solved_score=cumulative,
            upper_bound=bound,
            optimal_proven=optimal_proven,
            solver_stats=dict(solver_stats or {}),
            root_evaluations=root_evaluations,
            min_group=active_rules.min_group,
            clear_board_bonus=active_rules.clear_board_bonus,
        )

    @property
    def move_count(self) -> int:
        return len(self.steps)

    @property
    def leftover_blocks(self) -> int:
        if not self.steps:
            return self.initial_board.block_count
        return self.steps[-1].after_board.block_count

    @property
    def blocks_cleared(self) -> int:
        return self.initial_board.block_count - self.leftover_blocks

    @property
    def first_step(self) -> PlanStep | None:
        return self.steps[0] if self.steps else None

    @property
    def status(self) -> str:
        return "OPTIMAL PLAN PROVEN" if self.optimal_proven else "BEST KNOWN PLAN"

    @property
    def gap(self) -> int:
        return max(0, self.upper_bound - self.total_score)

    def replay_valid(self, *, rules: GameRules | None = None) -> bool:
        """Check the public replay invariant, including score conservation."""

        active_rules = rules or GameRules(
            min_group=self.min_group,
            clear_board_bonus=self.clear_board_bonus,
        )
        current = self.initial_board
        score = 0
        for expected_index, step in enumerate(self.steps, start=1):
            if step.index != expected_index or step.before_board != current:
                return False
            legal = find_groups(current, min_group=active_rules.min_group)
            if not any(
                candidate.color == step.move.color and candidate.cells == step.move.cells
                for candidate in legal
            ):
                return False
            next_board = apply_move(current, step.move, min_group=active_rules.min_group)
            immediate = active_rules.score_move(step.move, cleared_board=next_board.block_count == 0)
            score += immediate
            if next_board != step.after_board or immediate != step.immediate_score:
                return False
            if step.cumulative_score != score or step.group_size != step.move.size:
                return False
            current = next_board
        return score == self.total_score and is_terminal(current, min_group=active_rules.min_group)

    def summary(self) -> dict[str, object]:
        return {
            "total_score": self.total_score,
            "moves": self.move_count,
            "blocks_cleared": self.blocks_cleared,
            "leftover_blocks": self.leftover_blocks,
            "upper_bound": self.upper_bound,
            "gap": self.gap,
            "optimal_proven": self.optimal_proven,
            "status": self.status,
        }
