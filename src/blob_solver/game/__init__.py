"""Pure SameGame board model and deterministic transitions."""

from .board import Board, BoardInvariantError, Cell, Color
from .groups import Move, find_groups
from .rules import GameRules, score_move
from .transition import IllegalMoveError, apply_move, is_terminal

__all__ = [
    "Board",
    "BoardInvariantError",
    "Cell",
    "Color",
    "GameRules",
    "Move",
    "IllegalMoveError",
    "apply_move",
    "find_groups",
    "is_terminal",
    "score_move",
]
