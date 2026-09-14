"""Settled-board detection based on repeated identical valid observations."""

from __future__ import annotations

from dataclasses import dataclass
from time import monotonic, sleep
from typing import Callable

from blob_solver.game.board import Board

from .classifier import BoardObservation


class SettleTimeout(TimeoutError):
    pass


@dataclass(frozen=True, slots=True)
class SettledObservation:
    observation: BoardObservation
    elapsed_seconds: float
    frames: int

    @property
    def board(self) -> Board:
        if self.observation.board is None:
            raise ValueError("settled observation is not valid")
        return self.observation.board


class BoardSettler:
    def __init__(self, *, settle_frames: int = 3, interval_seconds: float = 0.08, timeout_seconds: float = 3.0) -> None:
        if settle_frames < 1 or interval_seconds <= 0 or timeout_seconds <= 0:
            raise ValueError("settle parameters must be positive")
        self.settle_frames = settle_frames
        self.interval_seconds = interval_seconds
        self.timeout_seconds = timeout_seconds

    def wait_for_settled(self, capture_observation: Callable[[], BoardObservation]) -> SettledObservation:
        started = monotonic()
        previous: Board | None = None
        consecutive = 0
        frames = 0
        last_observation: BoardObservation | None = None
        while monotonic() - started <= self.timeout_seconds:
            observation = capture_observation()
            frames += 1
            last_observation = observation
            board = observation.board if observation.valid else None
            if board is not None and board == previous:
                consecutive += 1
            elif board is not None:
                previous = board
                consecutive = 1
            else:
                previous = None
                consecutive = 0
            if board is not None and consecutive >= self.settle_frames:
                return SettledObservation(observation, monotonic() - started, frames)
            sleep(self.interval_seconds)
        detail = "no valid stable board"
        if last_observation is not None and last_observation.error:
            detail = last_observation.error
        raise SettleTimeout(f"board did not settle within {self.timeout_seconds:.2f}s: {detail}")
