"""Search algorithms for SameGame boards."""

from .base import SearchStats, Solution, Solver
from .beam import BeamConfig, BeamSolver, LegacyBeamSolver
from .exact import ExactSolver
from .greedy import GreedySolver
from .hybrid import HybridConfig, HybridPlanner, HybridPreset, SearchProgress
from .plan import Plan, PlanInvariantError, PlanStep, RootMoveEvaluation
from .rollout import RolloutConfig, RolloutSolver

__all__ = [
    "BeamConfig",
    "BeamSolver",
    "LegacyBeamSolver",
    "ExactSolver",
    "GreedySolver",
    "HybridConfig",
    "HybridPlanner",
    "HybridPreset",
    "Plan",
    "PlanInvariantError",
    "PlanStep",
    "RootMoveEvaluation",
    "SearchProgress",
    "RolloutConfig",
    "RolloutSolver",
    "SearchStats",
    "Solution",
    "Solver",
]
