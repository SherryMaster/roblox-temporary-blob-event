"""Computer-vision assisted SameGame solver."""

from .game.board import Board, Cell, Color
from .game.groups import Move, find_groups
from .game.rules import GameRules, score_move
from .game.transition import apply_move, is_terminal

__all__ = [
    "Board",
    "Cell",
    "Color",
    "GameRules",
    "Move",
    "apply_move",
    "find_groups",
    "is_terminal",
    "score_move",
]
