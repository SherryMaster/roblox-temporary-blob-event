"""Anytime beam search over several alternative SameGame continuations."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Event

from blob_solver.game.board import Board
from blob_solver.game.groups import Move, find_groups
from blob_solver.game.rules import GameRules
from blob_solver.game.transition import apply_move

from .base import SearchBudget, SearchStats, Solution, finish_solution, score_transition
from .greedy import GreedySolver
from .heuristics import color_ideal_upper_bound, evaluate_board, ordered_moves
from .transposition import TranspositionTable


@dataclass(frozen=True, slots=True)
class BeamConfig:
    beam_width: int = 5000
    max_nodes: int = 250_000
    candidate_moves: int | None = 80
    min_group: int = 2

    def __post_init__(self) -> None:
        if self.beam_width < 1 or self.max_nodes < 1:
            raise ValueError("beam_width and max_nodes must be positive")
        if self.candidate_moves is not None and self.candidate_moves < 1:
            raise ValueError("candidate_moves must be positive or None")
        if self.min_group < 2:
            raise ValueError("min_group must be at least 2")


@dataclass(frozen=True, slots=True)
class _BeamNode:
    board: Board
    score: int
    moves: tuple[Move, ...]


def _legacy_evaluate_board(board: Board, *, min_group: int = 2) -> float:
    """Pre-overhaul ordering heuristic used only by LegacyBeamSolver."""

    groups = find_groups(board, min_group=min_group)
    counts = board.color_counts()
    by_color: dict[object, list[int]] = {color: [] for color in counts}
    for group in groups:
        by_color.setdefault(group.color, []).append(group.size)
    removable = sum(size * size for sizes in by_color.values() for size in sizes)
    largest = max((group.size for group in groups), default=0)
    fragmentation_gap = 0
    singleton_count = 0
    for color, count in counts.items():
        sizes = by_color.get(color, [])
        fragmentation_gap += count * count - sum(size * size for size in sizes)
        singleton_count += count - sum(sizes)
    return (
        0.22 * color_ideal_upper_bound(board)
        + 0.30 * removable
        + 0.20 * (largest * largest)
        - 0.22 * fragmentation_gap
        - 0.10 * singleton_count
    )


def _legacy_ordered_moves(board: Board, *, min_group: int = 2) -> tuple[Move, ...]:
    candidates = find_groups(board, min_group=min_group)

    def key(move: Move) -> tuple[float, ...]:
        next_board = apply_move(board, move, validate=False)
        removed_columns = board.width - next_board.width
        features = _legacy_features(next_board, min_group=min_group)
        return (
            _legacy_evaluate_board(next_board, min_group=min_group) + 0.45 * move.immediate_score,
            float(move.immediate_score),
            float(removed_columns) * 0.06,
            features["largest"],
            -features["singletons"],
        )

    return tuple(sorted(candidates, key=key, reverse=True))


def _legacy_features(board: Board, *, min_group: int = 2) -> dict[str, float]:
    groups = find_groups(board, min_group=min_group)
    counts = board.color_counts()
    by_color: dict[object, list[int]] = {color: [] for color in counts}
    for group in groups:
        by_color.setdefault(group.color, []).append(group.size)
    singleton_count = 0
    fragmentation_gap = 0
    for color, count in counts.items():
        sizes = by_color.get(color, [])
        fragmentation_gap += count * count - sum(size * size for size in sizes)
        singleton_count += count - sum(sizes)
    return {
        "largest": float(max((group.size for group in groups), default=0) ** 2),
        "singletons": float(singleton_count),
        "fragmentation_gap": float(fragmentation_gap),
    }


class BeamSolver:
    def __init__(self, config: BeamConfig | None = None, *, rules: GameRules | None = None) -> None:
        self.config = config or BeamConfig()
        self.rules = rules or GameRules(min_group=self.config.min_group)

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
            max_nodes=self.config.max_nodes,
            stats=stats,
        )
        # A complete incumbent is important when a tiny budget expires before a
        # beam layer reaches a terminal state.
        incumbent = GreedySolver(
            min_group=self.config.min_group,
            rules=self.rules,
        ).solve(board)
        best_moves = incumbent.moves
        best_score = incumbent.total_score
        beam: list[_BeamNode] = [_BeamNode(board=board, score=0, moves=())]
        table = TranspositionTable(max_entries=max(self.config.beam_width * 20, 10_000))
        completion_cache: dict[Board, Solution] = {}
        table.update_reaching(board, 0)

        while beam and not budget.expired():
            children: list[_BeamNode] = []
            for node in beam:
                if not budget.visit():
                    break
                legal = ordered_moves(node.board, min_group=self.config.min_group)
                if self.config.candidate_moves is not None:
                    legal = legal[: self.config.candidate_moves]
                if not legal:
                    if node.score > best_score:
                        best_score, best_moves = node.score, node.moves
                    continue
                for move in legal:
                    if budget.expired():
                        break
                    next_board = apply_move(node.board, move, validate=False)
                    next_score = node.score + score_transition(node.board, move, self.rules)
                    if not table.update_reaching(next_board, next_score):
                        continue
                    child = _BeamNode(
                        board=next_board,
                        score=next_score,
                        moves=node.moves + (move,),
                    )
                    children.append(child)
                    # Convert every explored partial child into a complete
                    # candidate immediately. The old implementation waited
                    # for a terminal beam layer, so a time-limited search was
                    # often indistinguishable from greedy play.
                    completion = completion_cache.get(next_board)
                    if completion is None:
                        completion = GreedySolver(
                            policy="score",
                            min_group=self.config.min_group,
                            rules=self.rules,
                        ).solve(next_board)
                        completion_cache[next_board] = completion
                    stats.complete_rollouts += 1
                    stats.rollout_nodes += completion.nodes_examined
                    candidate_score = next_score + completion.total_score
                    if candidate_score > best_score:
                        best_score = candidate_score
                        best_moves = child.moves + completion.moves
                        stats.best_complete_score = best_score
            if budget.expired():
                break
            children.sort(
                key=lambda node: (
                    node.score
                    + color_ideal_upper_bound(node.board, self.rules)
                    + 0.35 * evaluate_board(node.board, min_group=self.config.min_group)
                ),
                reverse=True,
            )
            beam = children[: self.config.beam_width]

        stats.interrupted = stats.interrupted or budget.expired()
        return finish_solution(
            moves=best_moves,
            score=best_score,
            stats=stats,
            solver_name=f"beam-{self.config.beam_width}",
            optimal_proven=False,
            upper_bound=color_ideal_upper_bound(board, self.rules),
        )


class LegacyBeamSolver:
    """The pre-overhaul beam retained as a benchmark baseline only.

    It is intentionally not used by the controller. Unlike ``BeamSolver``, it
    updates its incumbent only when a beam node itself reaches a terminal
    state, which makes the old greedy-collapse behavior measurable.
    """

    def __init__(self, config: BeamConfig | None = None, *, rules: GameRules | None = None) -> None:
        self.config = config or BeamConfig()
        self.rules = rules or GameRules(min_group=self.config.min_group)

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
            max_nodes=self.config.max_nodes,
            stats=stats,
        )
        incumbent = GreedySolver(min_group=self.config.min_group, rules=self.rules).solve(board)
        best_moves = incumbent.moves
        best_score = incumbent.total_score
        beam: list[_BeamNode] = [_BeamNode(board=board, score=0, moves=())]
        table = TranspositionTable(max_entries=max(self.config.beam_width * 20, 10_000))
        table.update_reaching(board, 0)

        while beam and not budget.expired():
            children: list[_BeamNode] = []
            for node in beam:
                if not budget.visit():
                    break
                legal = _legacy_ordered_moves(node.board, min_group=self.config.min_group)
                if self.config.candidate_moves is not None:
                    legal = legal[: self.config.candidate_moves]
                if not legal:
                    if node.score > best_score:
                        best_score, best_moves = node.score, node.moves
                    continue
                for move in legal:
                    if budget.expired():
                        break
                    next_board = apply_move(node.board, move, validate=False)
                    next_score = node.score + score_transition(node.board, move, self.rules)
                    if not table.update_reaching(next_board, next_score):
                        continue
                    child = _BeamNode(
                        board=next_board,
                        score=next_score,
                        moves=node.moves + (move,),
                    )
                    children.append(child)
                    if not find_groups(next_board, min_group=self.config.min_group):
                        stats.terminal_plans += 1
                        if next_score > best_score:
                            best_score, best_moves = next_score, child.moves
            if budget.expired():
                break
            children.sort(
                key=lambda node: node.score + _legacy_evaluate_board(node.board, min_group=self.config.min_group),
                reverse=True,
            )
            beam = children[: self.config.beam_width]

        stats.interrupted = stats.interrupted or budget.expired()
        return finish_solution(
            moves=best_moves,
            score=best_score,
            stats=stats,
            solver_name=f"legacy-beam-{self.config.beam_width}",
            optimal_proven=False,
            upper_bound=color_ideal_upper_bound(board, self.rules),
        )
