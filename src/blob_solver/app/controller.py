"""Application boundary for single-scan planning and frozen execution."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
from threading import Event
from typing import Any, Callable

from blob_solver.desktop.coordinates import (
    CoordinateDiagnostic,
    CoordinateError,
    CoordinateMapper,
    query_hyprland_monitors,
)
from blob_solver.desktop.factory import create_desktop_backend
from blob_solver.desktop.overlay import HighlightSpec, NullOverlay
from blob_solver.game.board import Board
from blob_solver.game.groups import Move, find_groups
from blob_solver.game.rules import GameRules
from blob_solver.game.transition import apply_move
from blob_solver.solver.base import Solution
from blob_solver.solver.beam import BeamConfig, BeamSolver
from blob_solver.solver.exact import ExactSolver
from blob_solver.solver.greedy import GreedySolver
from blob_solver.solver.hybrid import HybridConfig, HybridPlanner, SearchProgress
from blob_solver.solver.plan import Plan, PlanStep
from blob_solver.solver.rollout import RolloutConfig, RolloutSolver
from blob_solver.vision.calibration import CalibrationProfile, calibrate
from blob_solver.vision.classifier import BoardObservation, classify_samples
from blob_solver.vision.grid import GridSpec, sample_grid
from blob_solver.vision.region import Region
from blob_solver.vision.settle import BoardSettler

from .config import AppConfig, load_config, save_config
from .session import GameGeneration, PlanningSession
from .state_machine import AppState, InvalidStateTransition, transition


class VisionSafetyError(RuntimeError):
    """Raised when a visual or lifecycle operation is unsafe."""


@dataclass(frozen=True, slots=True)
class ActionResult:
    expected: Board
    actual: Board
    matched: bool
    observation: BoardObservation


def configure_session_logging(path: str | Path) -> logging.Logger:
    logger = logging.getLogger("blob_solver.session")
    logger.setLevel(logging.INFO)
    resolved = str(Path(path).expanduser())
    if not any(getattr(handler, "blob_solver_path", None) == resolved for handler in logger.handlers):
        Path(resolved).parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(resolved, encoding="utf-8")
        handler.blob_solver_path = resolved  # type: ignore[attr-defined]
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    return logger


class AutomationController:
    """Desktop-facing state owner; search and board transitions stay pure."""

    def __init__(
        self,
        config: AppConfig | None = None,
        *,
        config_path: str | Path | None = None,
        desktop: object | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.config_path = Path(config_path).expanduser() if config_path else None
        self.config = config or load_config(self.config_path)
        self.desktop = desktop or create_desktop_backend(self.config.desktop.backend)
        self.desktop.overlay = self.desktop.overlay or NullOverlay()  # type: ignore[attr-defined]
        self.logger = logger or configure_session_logging(self.config.log_file)
        self.region: Region | None = self.config.region
        self.profile: CalibrationProfile | None = None
        if self.config.calibration:
            try:
                self.profile = CalibrationProfile.from_dict(self.config.calibration)
            except (TypeError, ValueError):
                self.profile = None
        self.rules = GameRules(min_group=self.config.board.min_group)
        self.session = PlanningSession(region=self.region)
        self.current_observation: BoardObservation | None = None
        self.current_board: Board | None = None
        self.current_solution: Solution | None = None
        self.active_plan: Plan | None = None
        self.estimated_score = 0
        self.state = AppState.NO_REGION if self.region is None else AppState.READY_TO_SCAN
        self._stop_event = Event()
        self._autoplay_active = False
        self._mismatch_count = 0
        self.last_recommended_move: Move | None = None
        self.last_coordinate_diagnostic: CoordinateDiagnostic | None = None
        self.last_search_progress: SearchProgress | None = None
        self.settler = BoardSettler(
            settle_frames=self.config.vision.settle_frames,
            interval_seconds=self.config.vision.settle_interval_ms / 1000,
            timeout_seconds=self.config.vision.settle_timeout_seconds,
        )

    @property
    def grid(self) -> GridSpec:
        return GridSpec(self.config.board.rows, self.config.board.cols)

    @property
    def current_step(self) -> int:
        return self.session.current_step

    @property
    def step_count(self) -> int:
        return self.active_plan.move_count if self.active_plan else 0

    @property
    def quality(self) -> str:
        return self.config.solver.quality

    def _set_state(self, target: AppState) -> None:
        try:
            self.state = transition(self.state, target)
        except InvalidStateTransition as exc:
            raise VisionSafetyError(str(exc)) from exc

    def _event(self, name: str, **payload: object) -> None:
        self.logger.info(
            json.dumps(
                {"timestamp": datetime.now(timezone.utc).isoformat(), "event": name, **payload},
                sort_keys=True,
                default=str,
            )
        )

    @staticmethod
    def _board_hash(board: Board | None) -> str | None:
        if board is None:
            return None
        return hashlib.sha256(repr(board.columns).encode("utf-8")).hexdigest()[:16]

    def select_region(self, region: Region | None = None) -> Region:
        selected = region or self.desktop.selector.select_region()  # type: ignore[attr-defined]
        self.region = selected
        self.config.region = selected
        self.session.region = selected
        save_config(self.config, self.config_path)
        self._set_state(AppState.READY_TO_SCAN)
        self._event("region_selected", region=selected.to_dict())
        return selected

    def _require_region(self) -> Region:
        if self.region is None:
            raise VisionSafetyError("select or configure a board region first")
        return self.region

    def _capture_samples(self) -> tuple[Any, tuple[Any, ...]]:
        region = self._require_region()
        overlay = self.desktop.overlay  # type: ignore[attr-defined]
        # A scan replaces the current generation. Do not let the capture
        # context restore a stale highlight from the discarded plan after the
        # screenshot has been taken.
        overlay.hide()
        with overlay.hidden_during_capture():
            image = self.desktop.capture.capture(region)  # type: ignore[attr-defined]
            samples = sample_grid(
                image,
                region,
                self.grid,
                patch_ratio=self.config.vision.patch_ratio,
            )
        return image, samples

    def _classify_captured(
        self,
        samples: tuple[Any, ...],
        *,
        force_calibration: bool = False,
    ) -> tuple[CalibrationProfile, BoardObservation]:
        profile = (
            calibrate(samples, num_colors=self.config.board.num_colors)
            if force_calibration or self.profile is None
            else self.profile
        )
        observation = classify_samples(
            samples,
            self.grid,
            profile,
            confidence_threshold=self.config.vision.confidence_threshold,
        )
        self._event(
            "capture",
            board=observation.text(),
            valid=observation.valid,
            confidence=observation.min_confidence,
            state_hash=self._board_hash(observation.board),
            error=observation.error,
        )
        return profile, observation

    def _freeze_captured_generation(
        self,
        image: Any,
        profile: CalibrationProfile,
        observation: BoardObservation,
        *,
        reset_score: bool = True,
    ) -> BoardObservation:
        self.current_observation = observation
        if not observation.valid or observation.board is None:
            self.current_board = None
            self.active_plan = None
            self.current_solution = None
            self.session.discard_plan()
            self._set_state(AppState.ERROR)
            raise VisionSafetyError(self._observation_error(observation))
        self.profile = profile
        self.current_board = observation.board
        self.current_solution = None
        self.active_plan = None
        if reset_score:
            self.estimated_score = 0
        self.session.freeze_generation(
            GameGeneration(
                screenshot=image,
                board=observation.board,
                observation=observation,
                calibration=profile,
                region=self._require_region(),
            )
        )
        self.config.calibration = profile.to_dict()
        save_config(self.config, self.config_path)
        return observation

    def _capture_and_freeze(self, *, force_calibration: bool = False) -> BoardObservation:
        self._set_state(AppState.SCANNING)
        try:
            image, samples = self._capture_samples()
            profile, observation = self._classify_captured(samples, force_calibration=force_calibration)
            self._freeze_captured_generation(image, profile, observation)
            self._set_state(AppState.READY_TO_SCAN)
            return observation
        except Exception:
            if self.state == AppState.SCANNING:
                self._set_state(AppState.ERROR)
            raise

    def calibrate(self) -> BoardObservation:
        """Explicit calibration; it uses only its one captured image."""

        return self._capture_and_freeze(force_calibration=True)

    def scan(self) -> BoardObservation:
        """Explicit one-image scan, deriving calibration from that same image."""

        return self._capture_and_freeze()

    def _accept_observation(self, observation: BoardObservation, *, reset_score: bool = False) -> None:
        """Compatibility hook for adapters/tests that already have an observation."""

        self.current_observation = observation
        self.current_board = observation.board
        self.active_plan = None
        self.current_solution = None
        if reset_score:
            self.estimated_score = 0
        if observation.board is not None and self.profile is not None and self.region is not None:
            # Keep adapters that provide an already-classified observation
            # compatible with the generation/session model. Such a synthetic
            # generation has no original image, but it is still immutable and
            # safe for text-only planning; normal desktop scans always provide
            # the real screenshot through _freeze_captured_generation().
            self.session.freeze_generation(
                GameGeneration(
                    screenshot=None,
                    board=observation.board,
                    observation=observation,
                    calibration=self.profile,
                    region=self.region,
                )
            )
        else:
            self.session.discard_plan()

    @staticmethod
    def _observation_error(observation: BoardObservation) -> str:
        uncertain = [
            (cell.row, cell.col, round(cell.confidence, 3))
            for cell in observation.low_confidence_cells
        ]
        unknown = [(cell.row, cell.col) for cell in observation.unknown_cells]
        details = [
            part
            for part in (
                observation.error,
                f"uncertain={uncertain}" if uncertain else "",
                f"unknown={unknown}" if unknown else "",
            )
            if part
        ]
        return "; ".join(details) or "board recognition failed"

    def _make_solver(self, mode: str | None = None, *, quality: str | None = None):
        selected = (mode or self.config.solver.mode).lower()
        settings = self.config.solver
        if selected in {"hybrid", "fast", "balanced", "deep", "exhaustive", "quality"}:
            selected_quality = quality or (
                selected
                if selected in {"fast", "balanced", "deep", "exhaustive"}
                else settings.quality
            )
            return HybridPlanner(
                HybridConfig(
                    quality=selected_quality,
                    beam_width=settings.beam_width,
                    max_nodes=settings.max_nodes,
                    candidate_moves=settings.candidate_moves if settings.candidate_moves > 0 else None,
                    exact_endgame_blocks=settings.exact_endgame_blocks,
                    exact_legal_groups=settings.exact_legal_groups,
                    exact_work_limit=settings.exact_work_limit,
                    seed=settings.rollout_seed,
                    min_group=self.config.board.min_group,
                    process_count=settings.process_count,
                ),
                rules=self.rules,
            )
        if selected in {"greedy", "greedy-score", "greedy-size"}:
            policy = "size" if selected == "greedy-size" else "score"
            return GreedySolver(
                policy=policy,
                min_group=self.config.board.min_group,
                rules=self.rules,
            )
        if selected == "beam":
            return BeamSolver(
                BeamConfig(
                    beam_width=settings.beam_width,
                    max_nodes=settings.max_nodes or 250_000,
                    candidate_moves=settings.candidate_moves if settings.candidate_moves > 0 else None,
                    min_group=self.config.board.min_group,
                ),
                rules=self.rules,
            )
        if selected in {"exact", "developer-exact"}:
            return ExactSolver(
                max_nodes=settings.max_nodes,
                min_group=self.config.board.min_group,
                rules=self.rules,
            )
        if selected in {"rollout", "random"}:
            return RolloutSolver(
                RolloutConfig(
                    seed=settings.rollout_seed,
                    candidate_moves=max(1, settings.candidate_moves),
                    min_group=self.config.board.min_group,
                ),
                rules=self.rules,
            )
        raise ValueError(f"unknown solver mode: {selected}")

    def _accept_solution(self, solution: Solution, board: Board) -> Plan:
        plan = solution.to_plan(board, rules=self.rules)
        self.session.accept_plan(plan)
        self.active_plan = plan
        self.current_solution = solution
        self.current_board = board
        self.estimated_score = 0
        self.last_recommended_move = None
        return plan

    def _remember_progress(self, progress: SearchProgress) -> None:
        self.last_search_progress = progress

    def _solve_frozen(
        self,
        *,
        mode: str | None = None,
        quality: str | None = None,
        time_limit: float | None = None,
        progress_callback: Callable[[SearchProgress], None] | None = None,
    ) -> Solution:
        if self.current_board is None or self.session.generation is None:
            raise VisionSafetyError("there is no frozen valid board to solve")
        self._set_state(AppState.SEARCHING)
        solver = self._make_solver(mode, quality=quality)
        if time_limit is not None:
            selected_time = time_limit
        elif isinstance(solver, HybridPlanner):
            # Let the selected quality preset own its normal duration. A
            # caller can still pass an explicit limit for a bounded run.
            selected_time = solver.config.preset.default_time_limit
        else:
            selected_time = self.config.solver.time_limit_seconds
        kwargs: dict[str, object] = {
            "time_limit": selected_time,
            "cancel_event": self._stop_event,
        }
        if isinstance(solver, HybridPlanner):
            kwargs["progress_callback"] = progress_callback or self._remember_progress
        try:
            solution = solver.solve(self.current_board, **kwargs)  # type: ignore[arg-type]
            self._accept_solution(solution, self.current_board)
            self._event(
                "analysis",
                solver=solution.solver_name,
                state_hash=self._board_hash(self.current_board),
                score=solution.total_score,
                moves=solution.move_count,
                optimal_proven=solution.optimal_proven,
                upper_bound=solution.upper_bound,
                nodes=solution.nodes_examined,
                terminal_plans=solution.terminal_plans,
                search_time=solution.search_time_seconds,
            )
            self._set_state(AppState.PLAN_READY)
            return solution
        except Exception:
            if self.state == AppState.SEARCHING:
                self._set_state(AppState.ERROR)
            raise

    def scan_and_solve(
        self,
        *,
        quality: str | None = None,
        time_limit: float | None = None,
        progress_callback: Callable[[SearchProgress], None] | None = None,
    ) -> Plan:
        """Capture once, reconstruct once, then solve the frozen generation."""

        if self._autoplay_active:
            raise VisionSafetyError("cannot scan while autoplay is running")
        if self.state not in {
            AppState.READY_TO_SCAN,
            AppState.PLAN_READY,
            AppState.PAUSED,
            AppState.ERROR,
        }:
            raise VisionSafetyError(f"cannot scan from state {self.state.value}")
        self._stop_event.clear()
        self._capture_and_freeze()
        self._solve_frozen(
            quality=quality,
            time_limit=time_limit,
            progress_callback=progress_callback,
        )
        if self.active_plan is None:
            raise VisionSafetyError("planner returned no complete plan")
        return self.active_plan

    def analyze(
        self,
        *,
        mode: str | None = None,
        quality: str | None = None,
        time_limit: float | None = None,
        progress_callback: Callable[[SearchProgress], None] | None = None,
    ) -> Solution:
        """Solve the frozen board; only an absent board triggers one scan."""

        if not self._autoplay_active:
            self._stop_event.clear()
        if self.current_board is None or self.session.generation is None:
            self.scan_and_solve(
                quality=quality,
                time_limit=time_limit,
                progress_callback=progress_callback,
            )
            if self.current_solution is None:
                raise VisionSafetyError("planner produced no solution")
            return self.current_solution
        return self._solve_frozen(
            mode=mode,
            quality=quality,
            time_limit=time_limit,
            progress_callback=progress_callback,
        )

    def _step_spec(self, step: PlanStep) -> HighlightSpec:
        if self.active_plan is None:
            raise VisionSafetyError("there is no active complete plan")
        return HighlightSpec(
            region=self._require_region(),
            rows=self.config.board.rows,
            cols=self.config.board.cols,
            cells=step.move.cells,
            click_cell=step.click_cell,
            group_size=step.group_size,
            immediate_score=step.immediate_score,
            projected_total=self.active_plan.total_score,
            step=step.index,
            step_count=self.active_plan.move_count,
            color=str(step.move.color),
        )

    def show_step(self, index: int | None = None) -> HighlightSpec | None:
        """Show the next executable step; indexed calls are inspection-only unless current."""

        if self.active_plan is None:
            raise VisionSafetyError("scan & solve first; there is no frozen complete plan")
        if index is not None:
            inspected = self.session.view(index)
            if index != self.session.current_step:
                # A predicted future/past board is safe to inspect in the
                # control panel, but must never be drawn over the live game as
                # if it were the current physical state.
                self.desktop.overlay.hide()  # type: ignore[attr-defined]
                self._set_state(AppState.PLAN_READY)
                return None
            step = inspected
        else:
            step = self.session.step()
        if step is None:
            self.desktop.overlay.hide()  # type: ignore[attr-defined]
            self._set_state(AppState.PLAN_READY)
            return None
        spec = self._step_spec(step)
        self.desktop.overlay.show(spec)  # type: ignore[attr-defined]
        self.last_recommended_move = step.move
        self._set_state(AppState.SHOWING_STEP)
        self._event(
            "recommendation",
            step=step.index,
            step_count=self.active_plan.move_count,
            color=str(step.move.color),
            cells=step.move.cells,
            size=step.group_size,
            immediate_score=step.immediate_score,
            cumulative_score=step.cumulative_score,
            projected_total=self.active_plan.total_score,
        )
        return spec

    def show_next_move(self, solution: Solution | None = None) -> HighlightSpec | None:
        """Compatibility alias using the frozen plan cursor."""

        if solution is not None and self.active_plan is None:
            if self.current_board is None:
                raise VisionSafetyError("cannot show a solution without its initial board")
            self._accept_solution(solution, self.current_board)
        return self.show_step()

    def next_step(self) -> PlanStep | None:
        """Advance a manually completed predicted step without a capture."""

        if self.active_plan is None:
            raise VisionSafetyError("there is no active plan")
        current = self.session.step()
        if current is None:
            raise VisionSafetyError("the plan is already complete")
        self.desktop.overlay.hide()  # type: ignore[attr-defined]
        self.session.advance()
        self.current_board = current.after_board
        self.estimated_score = current.cumulative_score
        self.last_recommended_move = None
        self._set_state(AppState.PLAN_READY)
        return self.session.step()

    def previous_step(self) -> PlanStep | None:
        if self.active_plan is None:
            raise VisionSafetyError("there is no active plan")
        # This is an inspection action. It must not move the execution cursor
        # backward after a physical click, because doing so could cause a
        # later Next Step to replay an already executed move.
        self.desktop.overlay.hide()  # type: ignore[attr-defined]
        self.session.previous_view()
        self._set_state(AppState.PLAN_READY)
        return self.session.view()

    def _coordinate_mapper(self) -> CoordinateMapper | None:
        existing = getattr(self.desktop, "coordinate_mapper", None)
        if isinstance(existing, CoordinateMapper):
            return existing
        try:
            monitors = query_hyprland_monitors()
        except CoordinateError:
            return None
        return CoordinateMapper(monitors, input_space=self.config.desktop.input_space)

    def _click_point(self, cell: tuple[int, int]) -> tuple[int, int]:
        region = self._require_region()
        screen_row = self.config.board.rows - 1 - cell[0]
        if not (
            0 <= screen_row < self.config.board.rows
            and 0 <= cell[1] < self.config.board.cols
        ):
            raise VisionSafetyError("recommended logical cell is outside configured screen geometry")
        mapper = self._coordinate_mapper()
        if mapper is not None and mapper.monitors:
            diagnostic = mapper.click_diagnostic(
                region,
                rows=self.config.board.rows,
                cols=self.config.board.cols,
                screen_row=screen_row,
                col=cell[1],
            )
            self.last_coordinate_diagnostic = diagnostic
            self._event("coordinate_diagnostic", **diagnostic.as_dict())
            return diagnostic.input_click_point
        return region.cell_center(self.config.board.rows, self.config.board.cols, screen_row, cell[1])

    def _execute_frozen_step(self, *, wait: bool = True) -> PlanStep:
        if self.active_plan is None:
            raise VisionSafetyError("scan & solve first; there is no frozen complete plan")
        step = self.session.step()
        if step is None:
            raise VisionSafetyError("the plan is already complete")
        self.show_step()
        if wait and self._stop_event.wait(self.config.automation.click_delay_ms / 1000):
            raise VisionSafetyError("execution stopped before click")
        x, y = self._click_point(step.click_cell)
        self.desktop.overlay.hide()  # type: ignore[attr-defined]
        self.desktop.input.click(x, y)  # type: ignore[attr-defined]
        if wait and self._stop_event.wait(self.config.automation.animation_delay_ms / 1000):
            raise VisionSafetyError("execution stopped during animation delay")
        self.session.advance()
        self.current_board = step.after_board
        self.estimated_score = step.cumulative_score
        self.last_recommended_move = None
        return step

    def click_current_recommendation(self) -> ActionResult | PlanStep:
        """Click a frozen step; readback occurs only in explicit Verified mode."""

        if self.config.automation.verified_execution:
            return self._verified_click_current()
        return self._execute_frozen_step()

    def _capture_observation(self, profile: CalibrationProfile | None = None) -> BoardObservation:
        region = self._require_region()
        active_profile = profile or self.profile
        if active_profile is None:
            raise VisionSafetyError("no calibration profile; use Scan & Solve")
        overlay = self.desktop.overlay  # type: ignore[attr-defined]
        with overlay.hidden_during_capture():
            image = self.desktop.capture.capture(region)  # type: ignore[attr-defined]
            samples = sample_grid(
                image,
                region,
                self.grid,
                patch_ratio=self.config.vision.patch_ratio,
            )
        observation = classify_samples(
            samples,
            self.grid,
            active_profile,
            confidence_threshold=self.config.vision.confidence_threshold,
        )
        self._event(
            "verified_capture",
            board=observation.text(),
            valid=observation.valid,
            confidence=observation.min_confidence,
            error=observation.error,
        )
        return observation

    def _capture_settled(self):
        return self.settler.wait_for_settled(self._capture_observation)

    def complete_action(self, move: Move, *, expected: Board | None = None) -> ActionResult:
        """Explicit readback path retained for Verified execution/recovery."""

        if self.current_board is None:
            raise VisionSafetyError("cannot validate an action without a current board")
        predicted = expected or apply_move(self.current_board, move)
        self._set_state(AppState.RECOVERY)
        self.desktop.overlay.hide()  # type: ignore[attr-defined]
        settled = self._capture_settled()
        actual = settled.board
        matched = actual == predicted
        action_score = self.rules.score_move(move, cleared_board=predicted.block_count == 0)
        self._event(
            "action_result",
            expected_board=predicted.columns,
            actual_board=actual.columns,
            expected_state_hash=self._board_hash(predicted),
            actual_state_hash=self._board_hash(actual),
            matched=matched,
            settle_seconds=settled.elapsed_seconds,
            frames=settled.frames,
        )
        self.current_observation = settled.observation
        self.current_board = actual
        if matched:
            self.estimated_score += action_score
            self._mismatch_count = 0
        else:
            self._mismatch_count += 1
        self.current_solution = None
        self.active_plan = None
        # Verified execution observes a new physical generation after every
        # click. Rebase the session to that observation so the next explicit
        # solve can be accepted without pretending it belongs to the original
        # screenshot.
        if self.profile is not None:
            self.session.freeze_generation(
                GameGeneration(
                    screenshot=None,
                    board=actual,
                    observation=settled.observation,
                    calibration=self.profile,
                    region=self._require_region(),
                )
            )
        else:
            self.session.discard_plan()
        self.last_recommended_move = None
        self._set_state(AppState.READY_TO_SCAN)
        return ActionResult(predicted, actual, matched, settled.observation)

    def _verified_click_current(self) -> ActionResult:
        if self.current_solution is None or self.current_solution.first_move is None:
            self.analyze()
        if self.current_solution is None or self.current_solution.first_move is None or self.current_board is None:
            raise VisionSafetyError("there is no legal move")
        move = self.current_solution.first_move
        x, y = self._click_point(move.click_cell)
        self.desktop.overlay.hide()  # type: ignore[attr-defined]
        self.desktop.input.click(x, y)  # type: ignore[attr-defined]
        return self.complete_action(move, expected=apply_move(self.current_board, move))

    def stop_autoplay(self, *, emergency: bool = False) -> None:
        self._stop_event.set()
        if emergency:
            self.desktop.input.emergency_stop()  # type: ignore[attr-defined]
        self.desktop.overlay.hide()  # type: ignore[attr-defined]
        if self.state in {
            AppState.SCANNING,
            AppState.SEARCHING,
            AppState.PLAN_READY,
            AppState.SHOWING_STEP,
            AppState.AUTOPLAY,
        }:
            self._set_state(AppState.PAUSED)
        self._event("autoplay_stopped", emergency=emergency)

    def recheck_manual_move(self) -> ActionResult | None:
        """Legacy explicit readback; normal Next Step never calls this."""

        if self.last_recommended_move is None:
            raise VisionSafetyError("no highlighted move; use Next Step or Rescan / Recover")
        return self.complete_action(self.last_recommended_move)

    def rescan_recover(
        self,
        *,
        quality: str | None = None,
        time_limit: float | None = None,
        progress_callback: Callable[[SearchProgress], None] | None = None,
    ) -> Plan:
        """Explicitly discard the frozen plan and rebuild from a new capture."""

        if self._autoplay_active:
            raise VisionSafetyError("stop autoplay before recovery")
        self._set_state(AppState.RECOVERY)
        self.session.discard_plan()
        self.active_plan = None
        self.current_solution = None
        self.current_board = None
        self._capture_and_freeze()
        self._solve_frozen(
            quality=quality,
            time_limit=time_limit,
            progress_callback=progress_callback,
        )
        if self.active_plan is None:
            raise VisionSafetyError("recovery planner returned no complete plan")
        return self.active_plan

    def _run_verified_autoplay(self) -> int:
        self._set_state(AppState.AUTOPLAY)
        moves_made = 0
        while not self._stop_event.is_set():
            if self.current_board is None or not find_groups(
                self.current_board,
                self.config.board.min_group,
            ):
                break
            solution = self.analyze()
            move = solution.first_move
            if move is None:
                break
            self.show_next_move(solution)
            if self._stop_event.wait(self.config.automation.click_delay_ms / 1000):
                break
            x, y = self._click_point(move.click_cell)
            self.desktop.overlay.hide()  # type: ignore[attr-defined]
            self.desktop.input.click(x, y)
            action = self.complete_action(move, expected=apply_move(self.current_board, move))
            if not action.matched:
                self._event("autoplay_safety_stop", mismatches=self._mismatch_count)
                break
            moves_made += 1
        return moves_made

    def run_autoplay(self) -> int:
        """Execute the already solved sequence without capture or re-solving."""

        self._stop_event.clear()
        try:
            if self.config.automation.verified_execution:
                if self.active_plan is None or self.current_board is None:
                    raise VisionSafetyError("scan & solve first; autoplay requires a frozen complete plan")
                self._autoplay_active = True
                return self._run_verified_autoplay()
            if self.active_plan is None:
                raise VisionSafetyError("scan & solve first; autoplay requires a frozen complete plan")
            self._autoplay_active = True
            self._set_state(AppState.AUTOPLAY)
            moves_made = 0
            while (
                not self._stop_event.is_set()
                and self.session.current_step < self.active_plan.move_count
            ):
                self._execute_frozen_step()
                moves_made += 1
            if self._stop_event.is_set():
                self._set_state(AppState.PAUSED)
            else:
                self._set_state(AppState.PLAN_READY)
            return moves_made
        except Exception as exc:
            self.desktop.overlay.hide()  # type: ignore[attr-defined]
            if not isinstance(exc, VisionSafetyError):
                self.state = AppState.ERROR
            self._event("autoplay_error", error=str(exc))
            raise
        finally:
            self.desktop.overlay.hide()  # type: ignore[attr-defined]
            self._autoplay_active = False
