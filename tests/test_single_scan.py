from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from blob_solver.app.config import AppConfig, BoardConfig, SolverConfig, VisionConfig
from blob_solver.app.controller import AutomationController, VisionSafetyError
from blob_solver.desktop.input import SafeClicker
from blob_solver.desktop.overlay import NullOverlay
from blob_solver.game.board import Board
from blob_solver.vision.capture import MemoryCapture
from blob_solver.vision.fixtures import make_board_image
from blob_solver.vision.region import Region


class FakeInput:
    def __init__(self) -> None:
        self.clicks: list[tuple[int, int]] = []

    def click(self, x: int, y: int) -> None:
        self.clicks.append((x, y))

    def emergency_stop(self) -> None:
        pass


def _controller(tmp_path):
    image = make_board_image([["red", "yellow"], ["red", "yellow"]], cell_size=40)
    capture = MemoryCapture(image)
    config = AppConfig(
        board=BoardConfig(rows=2, cols=2, min_group=2, num_colors=2),
        vision=VisionConfig(confidence_threshold=0.8),
        solver=SolverConfig(mode="hybrid", quality="fast", time_limit_seconds=0, max_nodes=2_000),
        region=Region(10, 20, 80, 80),
    )
    desktop = SimpleNamespace(
        capture=capture,
        selector=None,
        input=FakeInput(),
        overlay=NullOverlay(),
    )
    logger = logging.getLogger(f"blob_solver.single_scan.{id(capture)}")
    logger.addHandler(logging.NullHandler())
    controller = AutomationController(
        config,
        config_path=tmp_path / "config.toml",
        desktop=desktop,
        logger=logger,
    )
    return controller, capture, desktop.input


def test_default_scan_and_frozen_autoplay_capture_once(tmp_path) -> None:
    controller, capture, fake_input = _controller(tmp_path)

    controller.scan_and_solve(time_limit=0)
    moves = controller.run_autoplay()

    assert moves == 2
    assert capture.capture_count == 1
    assert len(fake_input.clicks) == moves
    assert controller.active_plan is not None
    assert controller.current_board is not None
    assert controller.current_board.block_count == 0


def test_autoplay_requires_an_accepted_frozen_plan(tmp_path) -> None:
    controller, capture, _ = _controller(tmp_path)

    with pytest.raises(VisionSafetyError, match="scan & solve first"):
        controller.run_autoplay()

    assert capture.capture_count == 0


def test_recovery_is_the_explicit_second_capture(tmp_path) -> None:
    controller, capture, _ = _controller(tmp_path)

    controller.scan_and_solve(time_limit=0)
    assert capture.capture_count == 1
    controller.rescan_recover(time_limit=0)
    assert capture.capture_count == 2


def test_previous_step_only_changes_predicted_view_not_execution_cursor(tmp_path) -> None:
    controller, capture, fake_input = _controller(tmp_path)
    controller.config.automation.click_delay_ms = 0
    controller.config.automation.animation_delay_ms = 0
    controller.scan_and_solve(time_limit=0)

    controller.show_step()
    first = controller.click_current_recommendation()
    assert first.index == 1
    assert controller.current_step == 1
    board_after_first = controller.current_board

    inspected = controller.previous_step()
    assert inspected is not None and inspected.index == 1
    assert controller.current_step == 1
    assert controller.current_board == board_after_first

    # Next Step still advances past step 2 rather than replaying the inspected
    # step; manual mode assumes the user performed that physical click.
    assert controller.next_step() is None
    assert controller.current_step == 2
    assert len(fake_input.clicks) == 1
    assert capture.capture_count == 1
