"""Features that reward future color consolidation, not only large current groups."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from blob_solver.game.board import Board
from blob_solver.game.groups import Move, find_components, find_groups
from blob_solver.game.rules import GameRules
from blob_solver.game.transition import apply_move


@dataclass(frozen=True, slots=True)
class HeuristicWeights:
    ideal_color_value: float = 0.16
    removable_component_value: float = 0.24
    largest_group_value: float = 0.16
    fragmentation_penalty: float = 0.16
    singleton_penalty: float = 0.08
    column_collapse_bonus: float = 0.12
    merge_gain_value: float = 0.22
    near_merge_value: float = 0.16
    blocker_penalty: float = 0.05


def color_ideal_upper_bound(board: Board, rules: GameRules | None = None) -> int:
    """Admissible optimistic score: merge every remaining color perfectly."""

    rules = rules or GameRules()
    clear_bonus = rules.clear_board_bonus if board.block_count else 0
    return sum(count * count for count in board.color_counts().values()) + clear_bonus


@lru_cache(maxsize=200_000)
def component_features(board: Board, *, min_group: int = 2) -> dict[str, float]:
    groups = find_groups(board, min_group=min_group)
    components = find_components(board)
    counts = board.color_counts()
    by_color: dict[object, list[Move]] = {color: [] for color in counts}
    for component in components:
        by_color.setdefault(component.color, []).append(component)
    removable = sum(group.size * group.size for group in groups)
    largest = max((group.size for group in groups), default=0)
    fragmentation_gap = 0
    singleton_count = 0
    merge_gain = 0.0
    near_merge_gain = 0.0
    blocker_cells = 0.0
    separated_components = 0
    for color, count in counts.items():
        same_color = by_color.get(color, [])
        sizes = [component.size for component in same_color]
        fragmentation_gap += count * count - sum(size * size for size in sizes)
        singleton_count += sum(size == 1 for size in sizes)
        separated_components += max(0, len(same_color) - 1)
        for left_index, left in enumerate(same_color):
            for right in same_color[left_index + 1 :]:
                gain = float((left.size + right.size) ** 2 - left.size**2 - right.size**2)
                merge_gain += gain
                left_rows = (
                    min(cell[0] for cell in left.cells),
                    max(cell[0] for cell in left.cells),
                )
                right_rows = (
                    min(cell[0] for cell in right.cells),
                    max(cell[0] for cell in right.cells),
                )
                left_columns = tuple(cell[1] for cell in left.cells)
                right_columns = tuple(cell[1] for cell in right.cells)
                horizontal_gap = max(
                    0,
                    min(left_columns) - max(right_columns) - 1,
                    min(right_columns) - max(left_columns) - 1,
                )
                vertical_gap = max(
                    0,
                    min(left_rows) - max(right_rows) - 1,
                    min(right_rows) - max(left_rows) - 1,
                )
                distance = horizontal_gap + vertical_gap
                # A small Manhattan separation is a cheap proxy for a merge
                # that gravity/column compaction may expose later.
                if distance <= 3:
                    near_merge_gain += gain * (4 - distance) / 3
                blocker_cells += max(0, distance - 1)
    return {
        "ideal": float(color_ideal_upper_bound(board)),
        "removable": float(removable),
        "largest": float(largest * largest),
        "fragmentation_gap": float(fragmentation_gap),
        "singletons": float(singleton_count),
        "legal_moves": float(len(groups)),
        "merge_gain": merge_gain,
        "near_merge_gain": near_merge_gain,
        "blocker_cells": blocker_cells,
        "separated_components": float(separated_components),
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
        + weights.merge_gain_value * features["merge_gain"]
        + weights.near_merge_value * features["near_merge_gain"]
        - weights.blocker_penalty * features["blocker_cells"]
    )


@lru_cache(maxsize=400_000)
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
    before_features = component_features(board, min_group=min_group)
    features = component_features(next_board, min_group=min_group)
    future_value = evaluate_board(next_board, weights, min_group=min_group)
    merge_delta = features["near_merge_gain"] - before_features["near_merge_gain"]
    empty_column_bonus = float(removed_columns) * weights.column_collapse_bonus
    return (
        future_value
        + 0.45 * float(move.immediate_score)
        + weights.near_merge_value * merge_delta,
        float(move.immediate_score),
        empty_column_bonus,
        features["merge_gain"],
        features["largest"],
        -features["singletons"],
        -features["blocker_cells"],
    )


@lru_cache(maxsize=200_000)
def ordered_moves(
    board: Board,
    moves: tuple[Move, ...] | None = None,
    *,
    min_group: int = 2,
    weights: HeuristicWeights | None = None,
) -> tuple[Move, ...]:
    candidates = find_groups(board, min_group=min_group) if moves is None else moves
    return tuple(
        sorted(
            candidates,
            key=lambda move: move_order_key(board, move, weights, min_group=min_group),
            reverse=True,
        )
    )
