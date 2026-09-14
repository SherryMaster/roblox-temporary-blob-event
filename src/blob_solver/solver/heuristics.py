"""Features that reward future color consolidation, not only large current groups."""

from __future__ import annotations

from dataclasses import dataclass

from blob_solver.game.board import Board
from blob_solver.game.groups import Move, find_groups
from blob_solver.game.rules import GameRules
from blob_solver.game.transition import apply_move


@dataclass(frozen=True, slots=True)
class HeuristicWeights:
    ideal_color_value: float = 0.22
    removable_component_value: float = 0.30
    largest_group_value: float = 0.20
    fragmentation_penalty: float = 0.22
    singleton_penalty: float = 0.10
    column_collapse_bonus: float = 0.06


def color_ideal_upper_bound(board: Board, rules: GameRules | None = None) -> int:
    """Admissible optimistic score: merge every remaining color perfectly."""

    rules = rules or GameRules()
    clear_bonus = rules.clear_board_bonus if board.block_count else 0
    return sum(count * count for count in board.color_counts().values()) + clear_bonus


def component_features(board: Board, *, min_group: int = 2) -> dict[str, float]:
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
        # Count cells in singleton components by subtracting legal components.
        singleton_count += count - sum(sizes)
    return {
        "ideal": float(color_ideal_upper_bound(board)),
        "removable": float(removable),
        "largest": float(largest * largest),
        "fragmentation_gap": float(fragmentation_gap),
        "singletons": float(singleton_count),
        "legal_moves": float(len(groups)),
    }


def evaluate_board(
    board: Board,
    weights: HeuristicWeights | None = None,
    *,
    min_group: int = 2,
) -> float:
    """Estimate future promise using color fragmentation and topology signals."""

    weights = weights or HeuristicWeights()
    features = component_features(board, min_group=min_group)
    return (
        weights.ideal_color_value * features["ideal"]
        + weights.removable_component_value * features["removable"]
        + weights.largest_group_value * features["largest"]
        - weights.fragmentation_penalty * features["fragmentation_gap"]
        - weights.singleton_penalty * features["singletons"]
    )


def move_order_key(
    board: Board,
    move: Move,
    weights: HeuristicWeights | None = None,
    *,
    min_group: int = 2,
) -> tuple[float, ...]:
    """Score a candidate with a cheap one-ply lookahead for search ordering."""

    weights = weights or HeuristicWeights()
    next_board = apply_move(board, move, validate=False)
    before_columns = board.width
    removed_columns = before_columns - next_board.width
    features = component_features(next_board, min_group=min_group)
    future_value = evaluate_board(next_board, weights, min_group=min_group)
    return (
        future_value + 0.45 * float(move.immediate_score),
        float(move.immediate_score),
        float(removed_columns) * weights.column_collapse_bonus,
        features["largest"],
        -features["singletons"],
    )


def ordered_moves(
    board: Board,
    moves: tuple[Move, ...] | None = None,
    *,
    min_group: int = 2,
) -> tuple[Move, ...]:
    candidates = find_groups(board, min_group=min_group) if moves is None else moves
    return tuple(
        sorted(
            candidates,
            key=lambda move: move_order_key(board, move, min_group=min_group),
            reverse=True,
        )
    )
