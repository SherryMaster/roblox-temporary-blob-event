from __future__ import annotations

import logging
from types import SimpleNamespace

from blob_solver.app.config import AppConfig, BoardConfig, SolverConfig, VisionConfig
from blob_solver.app.controller import AutomationController
from blob_solver.desktop.overlay import NullOverlay
from blob_solver.game.transition import apply_move
from blob_solver.vision.fixtures import make_board_image
from blob_solver.vision.region import Region


class SequenceCapture:
    def __init__(self, images) -> None:
        self.images = list(images)

    def capture(self, region):
        if len(self.images) > 1:
            return self.images.pop(0).copy()
        return self.images[0].copy()


class FakeInput:
    def __init__(self) -> None:
        self.clicks = []

    def click(self, x: int, y: int) -> None:
        self.clicks.append((x, y))

    def emergency_stop(self) -> None:
        pass


def test_controller_calibrates_recommends_and_validates_readback(tmp_path) -> None:
    initial_matrix = [["red", "yellow"], ["red", "yellow"]]
    initial_image = make_board_image(initial_matrix, cell_size=40)
    # The expected image is replaced below after the first observation is known.
    capture = SequenceCapture([initial_image])
    config = AppConfig(
        board=BoardConfig(rows=2, cols=2, min_group=2, num_colors=2),
        vision=VisionConfig(
            confidence_threshold=0.8,
            settle_frames=2,
            settle_interval_ms=1,
            settle_timeout_seconds=0.2,
        ),
        solver=SolverConfig(mode="exact", time_limit_seconds=1.0, max_nodes=1000),
        region=Region(10, 20, 80, 80),
    )
    desktop = SimpleNamespace(
        capture=capture,
        selector=None,
        input=FakeInput(),
        overlay=NullOverlay(),
    )
    logger = logging.getLogger("blob_solver.test_controller")
    logger.addHandler(logging.NullHandler())
    controller = AutomationController(config, config_path=tmp_path / "config.toml", desktop=desktop, logger=logger)

    observed = controller.calibrate()
    assert observed.valid
    solution = controller.analyze(mode="exact", time_limit=1)
    spec = controller.show_next_move(solution)
    assert spec is not None
    assert spec.group_size == 2

    assert controller.current_board is not None
    move = solution.first_move
    assert move is not None
    expected = apply_move(controller.current_board, move)
    expected_matrix = [list(row) for row in expected.to_matrix(rows=2, cols=2, empty=None)]
    expected_image = make_board_image(expected_matrix, cell_size=40)
    capture.images.extend([expected_image, expected_image])
    action = controller.recheck_manual_move()
    assert action is not None
    assert action.matched
    assert controller.estimated_score == move.immediate_score
    assert controller.last_recommended_move is None


def test_autoplay_stops_only_after_verified_sequence(tmp_path) -> None:
    initial_matrix = [["red", "yellow"], ["red", "yellow"]]
    initial_image = make_board_image(initial_matrix, cell_size=40)
    capture = SequenceCapture([initial_image])
    config = AppConfig(
        board=BoardConfig(rows=2, cols=2, min_group=2, num_colors=2),
        vision=VisionConfig(
            confidence_threshold=0.8,
            settle_frames=2,
            settle_interval_ms=1,
            settle_timeout_seconds=0.2,
        ),
        solver=SolverConfig(mode="exact", time_limit_seconds=1.0, max_nodes=1000),
        region=Region(10, 20, 80, 80),
    )
    fake_input = FakeInput()
    desktop = SimpleNamespace(
        capture=capture,
        selector=None,
        input=fake_input,
        overlay=NullOverlay(),
    )
    logger = logging.getLogger("blob_solver.test_autoplay")
    logger.addHandler(logging.NullHandler())
    controller = AutomationController(config, config_path=tmp_path / "config.toml", desktop=desktop, logger=logger)
    controller.calibrate()
    plan = controller.analyze(mode="exact", time_limit=1)
    assert controller.current_board is not None

    states = []
    current = controller.current_board
    for move in plan.moves:
        current = apply_move(current, move)
        states.append(current)
    capture.images = [
        make_board_image([list(row) for row in state.to_matrix(rows=2, cols=2, empty=None)], cell_size=40)
        for state in states
        for _ in range(2)
    ]
    config.automation.click_delay_ms = 0
    moves = controller.run_autoplay()
    assert moves == len(plan.moves)
    assert len(fake_input.clicks) == moves
    assert controller.current_board is not None and controller.current_board.block_count == 0
    assert controller.estimated_score == 8
