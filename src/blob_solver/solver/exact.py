"""Exact memoized branch-and-bound solving for small or late-game states."""

from __future__ import annotations

from threading import Event

from blob_solver.game.board import Board
from blob_solver.game.groups import Move
from blob_solver.game.rules import GameRules
from blob_solver.game.transition import apply_move

from .base import SearchBudget, SearchStats, Solution, finish_solution, score_transition
from .greedy import GreedySolver
from .heuristics import color_ideal_upper_bound, ordered_moves
from .transposition import TranspositionTable


class ExactSolver:
    """Prove the maximum score when the complete memoized search finishes.

    The future value of a board is independent of the path used to reach it.
    Since every legal move removes at least two cells, the state graph is
    acyclic by block count. Fully explored states are therefore safe to cache
    with both their exact future score and principal next move.
    """

    def __init__(
        self,
        *,
        max_nodes: int | None = None,
        min_group: int = 2,
        rules: GameRules | None = None,
    ) -> None:
        self.max_nodes = max_nodes
        if min_group < 2:
            raise ValueError("min_group must be at least 2")
        self.min_group = min_group
        self.rules = rules or GameRules(min_group=min_group)
        self.transposition_table: TranspositionTable | None = None

    def _reconstruct(self, board: Board, table: TranspositionTable) -> list[Move]:
        moves: list[Move] = []
        current = board
        while True:
            record = table.exact(current)
            if record is None or record.best_move is None:
                return moves
            move = record.best_move
            moves.append(move)
            current = apply_move(current, move, min_group=self.min_group, validate=False)

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
            max_nodes=self.max_nodes,
            stats=stats,
        )
        # Always build a complete incumbent before consulting a zero/expired
        # budget. This is also the cancellation safety guarantee.
        incumbent = GreedySolver(
            min_group=self.min_group,
            rules=self.rules,
        ).solve(board)
        best_score = incumbent.total_score
        best_moves = list(incumbent.moves)
        table = TranspositionTable(max_entries=None)
        self.transposition_table = table

        def future_value(current: Board) -> tuple[int, Move | None] | None:
            cached = table.exact(current)
            if cached is not None:
                return cached.exact_future_score or 0, cached.best_move
            if not budget.visit():
                return None
            legal = ordered_moves(current, min_group=self.min_group)
            if not legal:
                stats.terminal_plans += 1
                table.store_exact(current, 0, None)
                return 0, None

            best_future = -1
            best_move: Move | None = None
            fully_solved = True
            for move in legal:
                if budget.expired():
                    fully_solved = False
                    stats.interrupted = True
                    break
                next_board = apply_move(current, move, min_group=self.min_group, validate=False)
                immediate = score_transition(current, move, self.rules)
                # A solved sibling gives a local lower bound. The same
                # admissible color bound safely discards a child that cannot
                # improve that lower bound without fabricating an exact value.
                if best_future >= 0 and immediate + color_ideal_upper_bound(next_board, self.rules) <= best_future:
                    continue
                child = future_value(next_board)
                if child is None:
                    fully_solved = False
                    break
                total = immediate + child[0]
                if total > best_future:
                    best_future = total
                    best_move = move
            if not fully_solved:
                return None
            if best_future < 0:
                # This is reachable only when every legal child was safely
                # pruned after a previous sibling established the value.
                raise RuntimeError("exact search lost its local lower bound")
            table.store_exact(current, best_future, best_move)
            return best_future, best_move

        result = future_value(board)
        proven = result is not None and not stats.interrupted and not budget.expired()
        if proven:
            best_score = result[0]
            best_moves = self._reconstruct(board, table)
        interrupted = stats.interrupted or budget.expired()
        stats.interrupted = interrupted
        return finish_solution(
            moves=best_moves,
            score=best_score,
            stats=stats,
            solver_name="exact-branch-and-bound",
            optimal_proven=proven,
            upper_bound=best_score if proven else color_ideal_upper_bound(board, self.rules),
        )
