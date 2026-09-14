"""Scoring and rule policy, kept separate for future game variants."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GameRules:
    """SameGame rules currently observed in the target browser game."""

    min_group: int = 2
    clear_board_bonus: int = 0

    def __post_init__(self) -> None:
        if self.min_group < 2:
            raise ValueError("SameGame legal moves must contain at least two cells")
        if self.clear_board_bonus < 0:
            raise ValueError("clear_board_bonus cannot be negative")

    def score_size(self, size: int, *, cleared_board: bool = False) -> int:
        """Return the score for a legal group size."""

        if size < self.min_group:
            raise ValueError(f"group of size {size} is not legal")
        return size * size + (self.clear_board_bonus if cleared_board else 0)

    def score_move(self, move: object, *, cleared_board: bool = False) -> int:
        """Score a move-like object exposing ``size`` or ``cells``."""

        if isinstance(move, int):
            size = move
        elif hasattr(move, "size"):
            size = int(getattr(move, "size"))
        else:
            size = len(getattr(move, "cells"))
        return self.score_size(size, cleared_board=cleared_board)

    def score_sequence(self, moves: object) -> int:
        return sum(self.score_move(move) for move in moves)  # type: ignore[union-attr]


def score_move(move: object) -> int:
    """Convenience function for the observed square scoring rule."""

    if isinstance(move, int):
        size = move
    elif hasattr(move, "size"):
        size = int(getattr(move, "size"))
    else:
        size = len(getattr(move, "cells"))
    if size < 2:
        raise ValueError("a legal SameGame move contains at least two cells")
    return size * size
