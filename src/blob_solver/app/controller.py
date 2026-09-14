"""Orchestrate capture, vision, search, hinting, and safe one-move-at-a-time play."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
from threading import Event

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
from blob_solver.solver.rollout import RolloutConfig, RolloutSolver
from blob_solver.vision.calibration import CalibrationProfile, calibrate
from blob_solver.vision.classifier import BoardObservation, classify_samples
from blob_solver.vision.grid import GridSpec, sample_grid
from blob_solver.vision.region import Region
from blob_solver.vision.settle import BoardSettler

from .config import AppConfig, load_config, save_config
from .state_machine import AppState


class VisionSafetyError(RuntimeError):
    """Raised when the visual board is not safe to use for automation."""


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
    """Stateful application boundary; the solver itself remains desktop-free."""

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
        self.rules = GameRules(min_group=self.config.board.min_group)
        if self.config.calibration:
            try:
                self.profile = CalibrationProfile.from_dict(self.config.calibration)
            except (TypeError, ValueError):
                self.profile = None
        self.current_observation: BoardObservation | None = None
        self.current_board: Board | None = None
        self.current_solution: Solution | None = None
        self.estimated_score = 0
        self.state = AppState.IDLE if self.region is None else AppState.REGION_SELECTED
        self._stop_event = Event()
        self._autoplay_active = False
        self._mismatch_count = 0
        self.last_recommended_move: Move | None = None
        self.settler = BoardSettler(
            settle_frames=self.config.vision.settle_frames,
            interval_seconds=self.config.vision.settle_interval_ms / 1000,
            timeout_seconds=self.config.vision.settle_timeout_seconds,
        )

    @property
    def grid(self) -> GridSpec:
        return GridSpec(self.config.board.rows, self.config.board.cols)

    def _event(self, name: str, **payload: object) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": name,
            **payload,
        }
        self.logger.info(json.dumps(record, sort_keys=True, default=str))

    @staticmethod
    def _board_hash(board: Board | None) -> str | None:
        if board is None:
            return None
        return hashlib.sha256(repr(board.columns).encode("utf-8")).hexdigest()[:16]

    def select_region(self, region: Region | None = None) -> Region:
        selected = region or self.desktop.selector.select_region()  # type: ignore[attr-defined]
        self.region = selected
        self.config.region = selected
        save_config(self.config, self.config_path)
        self.state = AppState.REGION_SELECTED
        self._event("region_selected", region=selected.to_dict())
        return selected

    def _require_region(self) -> Region:
        if self.region is None:
            raise VisionSafetyError("select or configure a board region first")
        return self.region

    def _capture_observation(self, profile: CalibrationProfile | None = None) -> BoardObservation:
        region = self._require_region()
        active_profile = profile or self.profile
        if active_profile is None:
            raise VisionSafetyError("calibrate before scanning")
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
            "capture",
            board=observation.text(),
            valid=observation.valid,
            confidence=observation.min_confidence,
            state_hash=self._board_hash(observation.board),
            error=observation.error,
        )
        return observation

    def calibrate(self) -> BoardObservation:
        self.state = AppState.CALIBRATING
        region = self._require_region()
        overlay = self.desktop.overlay  # type: ignore[attr-defined]
        with overlay.hidden_during_capture():
            image = self.desktop.capture.capture(region)  # type: ignore[attr-defined]
            samples = sample_grid(
                image,
                region,
                self.grid,
                patch_ratio=self.config.vision.patch_ratio,
            )
        self.profile = calibrate(samples, num_colors=self.config.board.num_colors)
        observation = classify_samples(
            samples,
            self.grid,
            self.profile,
            confidence_threshold=self.config.vision.confidence_threshold,
        )
        self._accept_observation(observation, reset_score=True)
        if not observation.valid:
            self.state = AppState.ERROR
            raise VisionSafetyError(self._observation_error(observation))
        self.config.calibration = self.profile.to_dict()
        save_config(self.config, self.config_path)
        self._event("calibrated", profile=self.profile.to_dict(), confidence=observation.min_confidence)
        self.state = AppState.READY
        return observation

    def scan(self) -> BoardObservation:
        observation = self._capture_observation()
        self._accept_observation(observation)
        if not observation.valid:
            self.state = AppState.ERROR
            raise VisionSafetyError(self._observation_error(observation))
        self.state = AppState.READY
        return observation

    def _accept_observation(self, observation: BoardObservation, *, reset_score: bool = False) -> None:
        self.current_observation = observation
        if observation.board is not None:
            self.current_board = observation.board
        if reset_score:
            self.estimated_score = 0
        self.current_solution = None

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

    def _make_solver(self, mode: str | None = None):
        selected = (mode or self.config.solver.mode).lower()
        settings = self.config.solver
        if selected in {"greedy", "greedy-score", "greedy-size"}:
            policy = "size" if selected == "greedy-size" else "score"
            return GreedySolver(
                policy=policy,
                min_group=self.config.board.min_group,
                rules=self.rules,
            )
        if selected == "beam":
            candidate_moves = settings.candidate_moves if settings.candidate_moves > 0 else None
            return BeamSolver(
                BeamConfig(
                    beam_width=settings.beam_width,
                    max_nodes=settings.max_nodes,
                    candidate_moves=candidate_moves,
                    min_group=self.config.board.min_group,
                ),
                rules=self.rules,
            )
        if selected in {"exact", "deep"}:
            return ExactSolver(
                max_nodes=settings.max_nodes,
                min_group=self.config.board.min_group,
                rules=self.rules,
            )
        if selected in {"rollout", "random", "mcts"}:
            return RolloutSolver(
                RolloutConfig(
                    seed=settings.rollout_seed,
                    candidate_moves=max(1, settings.candidate_moves),
                    min_group=self.config.board.min_group,
                ),
                rules=self.rules,
            )
        raise ValueError(f"unknown solver mode: {selected}")

    def analyze(self, *, mode: str | None = None, time_limit: float | None = None) -> Solution:
        if not self._autoplay_active:
            self._stop_event.clear()
        if self.current_board is None:
            self.scan()
        if self.current_board is None:
            raise VisionSafetyError("there is no valid board to analyze")
        self.state = AppState.ANALYZING
        solver = self._make_solver(mode)
        selected_time = self.config.solver.time_limit_seconds if time_limit is None else time_limit
        solution = solver.solve(
            self.current_board,
            time_limit=selected_time,
            cancel_event=self._stop_event,
        )
        self.current_solution = solution
        self._event(
            "analysis",
            solver=solution.solver_name,
            state_hash=self._board_hash(self.current_board),
            score=solution.total_score,
            moves=solution.move_count,
            optimal_proven=solution.optimal_proven,
            upper_bound=solution.upper_bound,
            nodes=solution.nodes_examined,
            search_time=solution.search_time_seconds,
        )
        self.state = AppState.HINT_VISIBLE if solution.first_move else AppState.READY
        return solution

    def show_next_move(self, solution: Solution | None = None) -> HighlightSpec | None:
        active = solution or self.current_solution or self.analyze()
        move = active.first_move
        if move is None:
            self.desktop.overlay.hide()  # type: ignore[attr-defined]
            self.state = AppState.READY
            return None
        self.last_recommended_move = move
        region = self._require_region()
        spec = HighlightSpec(
            region=region,
            rows=self.config.board.rows,
            cols=self.config.board.cols,
            cells=move.cells,
            click_cell=move.click_cell,
            group_size=move.size,
            immediate_score=move.immediate_score,
            projected_total=active.total_score,
        )
        self.desktop.overlay.show(spec)  # type: ignore[attr-defined]
        self.state = AppState.HINT_VISIBLE
        self._event(
            "recommendation",
            color=str(move.color),
            cells=move.cells,
            size=move.size,
            immediate_score=move.immediate_score,
            projected_total=active.total_score,
        )
        return spec

    def _capture_settled(self):
        return self.settler.wait_for_settled(self._capture_observation)

    def complete_action(self, move: Move, *, expected: Board | None = None) -> ActionResult:
        """Read back one manually or automatically executed move and revalidate it."""

        if self.current_board is None:
            raise VisionSafetyError("cannot validate an action without a current board")
        predicted = expected or apply_move(self.current_board, move)
        self.state = AppState.WAITING_SETTLE
        self.desktop.overlay.hide()  # type: ignore[attr-defined]
        try:
            settled = self._capture_settled()
        except Exception as exc:
            self.state = AppState.ERROR
            self._event("settle_error", error=str(exc))
            raise
        actual = settled.board
        matched = actual == predicted
        action_score = self.rules.score_move(
            move,
            cleared_board=predicted.block_count == 0,
        )
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
        self.current_solution = None
        self.last_recommended_move = None
        if matched:
            self.estimated_score += action_score
            self._mismatch_count = 0
            self.state = AppState.READY
        else:
            self._mismatch_count += 1
            self.state = AppState.READY
            # Replan from actual visual state, but do not continue clicking
            # automatically. The caller decides whether to show the new hint.
            try:
                self.analyze()
            except Exception as exc:
                self._event("replan_failed", error=str(exc))
        return ActionResult(predicted, actual, matched, settled.observation)

    def click_current_recommendation(self) -> ActionResult:
        """Perform exactly one recommended click, then verify the physical state."""

        if self.current_solution is None or self.current_solution.first_move is None:
            self.analyze()
        if self.current_solution is None or self.current_solution.first_move is None:
            raise VisionSafetyError("there is no legal move")
        move = self.current_solution.first_move
        region = self._require_region()
        screen_row = self.config.board.rows - 1 - move.click_cell[0]
        if not (
            0 <= screen_row < self.config.board.rows
            and 0 <= move.click_cell[1] < self.config.board.cols
        ):
            raise VisionSafetyError("recommended logical cell is outside configured screen geometry")
        x, y = region.cell_center(
            self.config.board.rows,
            self.config.board.cols,
            screen_row,
            move.click_cell[1],
        )
        self.desktop.overlay.hide()  # type: ignore[attr-defined]
        self.desktop.input.click(x, y)  # type: ignore[attr-defined]
        expected = apply_move(self.current_board, move)  # type: ignore[arg-type]
        return self.complete_action(move, expected=expected)

    def stop_autoplay(self, *, emergency: bool = False) -> None:
        self._stop_event.set()
        if emergency:
            self.desktop.input.emergency_stop()  # type: ignore[attr-defined]
        self.desktop.overlay.hide()  # type: ignore[attr-defined]
        self.state = AppState.STOPPED
        self._event("autoplay_stopped", emergency=emergency)

    def recheck_manual_move(self) -> ActionResult | None:
        """Wait for and validate the last hint after the user clicked it."""

        if self.last_recommended_move is None:
            self.scan()
            return None
        return self.complete_action(self.last_recommended_move)

    def run_autoplay(self) -> int:
        """Run safe autoplay until terminal, stop, or a readback discrepancy."""

        self._stop_event.clear()
        self._mismatch_count = 0
        self._autoplay_active = True
        if self.current_board is None:
            try:
                self.scan()
            except Exception:
                self._autoplay_active = False
                raise
        if self.current_board is None:
            self._autoplay_active = False
            raise VisionSafetyError("cannot autoplay without a valid board")
        self.state = AppState.AUTOPLAY
        moves_made = 0
        try:
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
                region = self._require_region()
                screen_row = self.config.board.rows - 1 - move.click_cell[0]
                x, y = region.cell_center(
                    self.config.board.rows,
                    self.config.board.cols,
                    screen_row,
                    move.click_cell[1],
                )
                self.desktop.overlay.hide()  # type: ignore[attr-defined]
                self.desktop.input.click(x, y)  # type: ignore[attr-defined]
                expected = apply_move(self.current_board, move)
                action = self.complete_action(move, expected=expected)
                if not action.matched:
                    # The result was re-read and replanned, but no additional
                    # physical input is permitted after a discrepancy.
                    self._event("autoplay_safety_stop", mismatches=self._mismatch_count)
                    break
                moves_made += 1
            if self._stop_event.is_set():
                self.state = AppState.STOPPED
            elif self.state != AppState.ERROR:
                self.state = AppState.READY
            return moves_made
        except Exception as exc:
            self.desktop.overlay.hide()  # type: ignore[attr-defined]
            self.state = AppState.ERROR
            self._event("autoplay_error", error=str(exc))
            raise
        finally:
            self.desktop.overlay.hide()  # type: ignore[attr-defined]
            self._autoplay_active = False
