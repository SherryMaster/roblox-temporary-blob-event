"""Explicit single-scan generation and frozen-plan session state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from blob_solver.game.board import Board
from blob_solver.solver.plan import Plan
from blob_solver.vision.calibration import CalibrationProfile
from blob_solver.vision.classifier import BoardObservation
from blob_solver.vision.region import Region


@dataclass(frozen=True, slots=True)
class GameGeneration:
    """The one physical observation that defines a deterministic generation."""

    screenshot: Any
    board: Board
    observation: BoardObservation
    calibration: CalibrationProfile
    region: Region


@dataclass(frozen=True, slots=True)
class SolvedGeneration:
    generation: GameGeneration
    plan: Plan


@dataclass(slots=True)
class PlanningSession:
    """Own the frozen generation, solution, fixed region, and plan cursor."""

    region: Region | None = None
    generation: GameGeneration | None = None
    solved_generation: SolvedGeneration | None = None
    current_step: int = 0
    view_step: int = 0

    @property
    def plan(self) -> Plan | None:
        return self.solved_generation.plan if self.solved_generation else None

    @property
    def board(self) -> Board | None:
        return self.generation.board if self.generation else None

    def freeze_generation(self, generation: GameGeneration) -> None:
        self.region = generation.region
        self.generation = generation
        self.solved_generation = None
        self.current_step = 0
        self.view_step = 0

    def accept_plan(self, plan: Plan) -> None:
        if self.generation is None:
            raise ValueError("cannot accept a plan without a frozen generation")
        if plan.initial_board != self.generation.board:
            raise ValueError("plan does not belong to the frozen generation")
        if not plan.replay_valid():
            raise ValueError("cannot accept an invalid plan")
        self.solved_generation = SolvedGeneration(self.generation, plan)
        self.current_step = 0
        self.view_step = 0

    def step(self, index: int | None = None):
        plan = self.plan
        if plan is None:
            return None
        if index is not None:
            if not 0 <= index < len(plan.steps):
                raise IndexError("plan step outside the complete plan")
            self.current_step = index
            self.view_step = index
        if not plan.steps or self.current_step >= len(plan.steps):
            return None
        return plan.steps[self.current_step]

    def view(self, index: int | None = None):
        """Select a predicted step for inspection without changing execution."""

        plan = self.plan
        if plan is None or not plan.steps:
            return None
        selected = self.view_step if index is None else index
        if not 0 <= selected < len(plan.steps):
            raise IndexError("plan view step outside the complete plan")
        self.view_step = selected
        return plan.steps[selected]

    def advance(self) -> None:
        if self.plan is None:
            raise ValueError("there is no active plan")
        if self.current_step >= len(self.plan.steps):
            raise ValueError("the plan is already complete")
        self.current_step += 1
        self.view_step = self.current_step

    def previous(self) -> None:
        if self.plan is None:
            raise ValueError("there is no active plan")
        self.current_step = max(0, self.current_step - 1)
        self.view_step = self.current_step

    def previous_view(self) -> None:
        """Move only the inspection cursor backward; physical execution is unchanged."""

        if self.plan is None:
            raise ValueError("there is no active plan")
        self.view_step = max(0, self.view_step - 1)

    def discard_plan(self) -> None:
        self.solved_generation = None
        self.current_step = 0
        self.view_step = 0
