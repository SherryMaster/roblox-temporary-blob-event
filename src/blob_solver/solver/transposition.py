"""Hash-based search caches shared by approximate and exact solvers."""

from __future__ import annotations

from dataclasses import dataclass

from blob_solver.game.board import Board
from blob_solver.game.groups import Move


@dataclass(frozen=True, slots=True)
class StateRecord:
    """Useful information known about a canonical board."""

    best_score_reaching: int = -1
    exact_future_score: int | None = None
    best_move: Move | None = None


class TranspositionTable:
    """A deliberately small, understandable transposition table.

    The approximate solvers use ``best_score_reaching`` to discard a weaker
    duplicate path. Exact search may additionally store a proven future value.
    """

    def __init__(self, max_entries: int | None = 500_000) -> None:
        self.max_entries = max_entries
        self._records: dict[Board, StateRecord] = {}

    def __len__(self) -> int:
        return len(self._records)

    def get(self, board: Board) -> StateRecord | None:
        return self._records.get(board)

    def update_reaching(
        self,
        board: Board,
        score: int,
        *,
        best_move: Move | None = None,
    ) -> bool:
        """Record a path if it improves the score reaching ``board``.

        Return true when the record changed. A bounded table evicts an arbitrary
        old entry only when its configured capacity is reached; correctness is
        unaffected because eviction only loses a performance optimization.
        """

        old = self._records.get(board)
        if old is not None and score <= old.best_score_reaching:
            return False
        if self.max_entries is not None and len(self._records) >= self.max_entries:
            self._records.pop(next(iter(self._records)))
        self._records[board] = StateRecord(
            best_score_reaching=score,
            exact_future_score=None if old is None else old.exact_future_score,
            best_move=best_move if best_move is not None else (None if old is None else old.best_move),
        )
        return True

    def store_exact(self, board: Board, score: int, best_move: Move | None) -> None:
        old = self._records.get(board)
        self._records[board] = StateRecord(
            best_score_reaching=-1 if old is None else old.best_score_reaching,
            exact_future_score=score,
            best_move=best_move,
        )

    def exact(self, board: Board) -> StateRecord | None:
        record = self._records.get(board)
        if record is None or record.exact_future_score is None:
            return None
        return record

    def clear(self) -> None:
        self._records.clear()
