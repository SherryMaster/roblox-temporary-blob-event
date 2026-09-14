"""Small responsive PySide6 control panel for hint mode and autoplay."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from blob_solver.game.groups import find_groups

from .config import load_config
from .controller import AutomationController, VisionSafetyError


def run_app(*, config_path: str | Path | None = None) -> int:
    try:
        from PySide6.QtWidgets import QApplication, QMessageBox
    except ImportError as exc:
        raise RuntimeError("install the ui extra with: pip install 'blob-solver[ui]'") from exc
    application = QApplication.instance() or QApplication([])
    try:
        window = MainWindow(config_path=config_path)
    except Exception as exc:
        QMessageBox.critical(None, "Blob Solver", str(exc))
        return 2
    window.show()
    return application.exec()


class MainWindow:
    """The UI imports Qt only after run_app, keeping CLI/core imports lightweight."""

    def __new__(cls, *, config_path: str | Path | None = None) -> Any:
        try:
            from PySide6.QtCore import QThread, Signal, Qt
            from PySide6.QtWidgets import (
                QComboBox,
                QFormLayout,
                QGridLayout,
                QGroupBox,
                QHBoxLayout,
                QLabel,
                QLineEdit,
                QMainWindow,
                QMessageBox,
                QPlainTextEdit,
                QPushButton,
                QSpinBox,
                QDoubleSpinBox,
                QVBoxLayout,
                QWidget,
            )
        except ImportError as exc:  # pragma: no cover - optional UI
            raise RuntimeError("PySide6 is required for the control panel") from exc

        class TaskThread(QThread):
            result = Signal(object)
            failed = Signal(str)

            def __init__(self, task: Callable[[], object]) -> None:
                super().__init__()
                self.task = task

            def run(self) -> None:
                try:
                    self.result.emit(self.task())
                except Exception as exc:
                    self.failed.emit(str(exc))

        class Window(QMainWindow):
            def __init__(self) -> None:
                super().__init__()
                self.setWindowTitle("Blob Solver")
                self.resize(900, 700)
                self.config_path = config_path
                self.controller = AutomationController(
                    load_config(config_path),
                    config_path=config_path,
                )
                self.worker: TaskThread | None = None
                self._build()

            def _build(self) -> None:
                root = QWidget()
                self.setCentralWidget(root)
                layout = QVBoxLayout(root)

                board_box = QGroupBox("BOARD")
                board_layout = QFormLayout(board_box)
                self.region_edit = QLineEdit()
                if self.controller.region:
                    region = self.controller.region
                    self.region_edit.setText(f"{region.x},{region.y},{region.width},{region.height}")
                self.rows = QSpinBox()
                self.rows.setRange(1, 1000)
                self.rows.setValue(self.controller.config.board.rows)
                self.cols = QSpinBox()
                self.cols.setRange(1, 1000)
                self.cols.setValue(self.controller.config.board.cols)
                select_button = QPushButton("Select Region")
                select_button.clicked.connect(self._select_region)
                calibrate_button = QPushButton("Calibrate")
                calibrate_button.clicked.connect(self._calibrate)
                scan_button = QPushButton("Scan Board")
                scan_button.clicked.connect(self._scan)
                region_row = QHBoxLayout()
                region_row.addWidget(self.region_edit)
                region_row.addWidget(select_button)
                board_layout.addRow("Region", region_row)
                board_layout.addRow("Rows", self.rows)
                board_layout.addRow("Columns", self.cols)
                board_actions = QHBoxLayout()
                board_actions.addWidget(calibrate_button)
                board_actions.addWidget(scan_button)
                board_layout.addRow(board_actions)
                layout.addWidget(board_box)

                solver_box = QGroupBox("SOLVER")
                solver_layout = QFormLayout(solver_box)
                self.mode = QComboBox()
                self.mode.addItems(["beam", "exact", "rollout", "greedy", "greedy-size"])
                self.mode.setCurrentText(self.controller.config.solver.mode)
                self.time_limit = QDoubleSpinBox()
                self.time_limit.setRange(0.0, 3600.0)
                self.time_limit.setDecimals(2)
                self.time_limit.setValue(self.controller.config.solver.time_limit_seconds)
                self.beam_width = QSpinBox()
                self.beam_width.setRange(1, 1_000_000)
                self.beam_width.setValue(self.controller.config.solver.beam_width)
                solver_layout.addRow("Mode", self.mode)
                solver_layout.addRow("Time budget (s)", self.time_limit)
                solver_layout.addRow("Beam width", self.beam_width)
                layout.addWidget(solver_box)

                action_box = QGroupBox("ACTION")
                action_layout = QGridLayout(action_box)
                analyze = QPushButton("Analyze")
                analyze.clicked.connect(self._analyze)
                show_hint = QPushButton("Show Next Move")
                show_hint.clicked.connect(self._show_hint)
                autoplay = QPushButton("Auto Play")
                autoplay.clicked.connect(self._autoplay)
                stop = QPushButton("Stop")
                stop.clicked.connect(self._stop)
                emergency = QPushButton("Emergency Stop")
                emergency.clicked.connect(self._emergency_stop)
                reread = QPushButton("Re-read After Manual Click")
                reread.clicked.connect(self._reread)
                for index, button in enumerate((analyze, show_hint, autoplay, stop, emergency, reread)):
                    action_layout.addWidget(button, index // 3, index % 3)
                layout.addWidget(action_box)

                self.status = QLabel("Ready")
                self.status.setWordWrap(True)
                layout.addWidget(self.status)
                self.debug = QPlainTextEdit()
                self.debug.setReadOnly(True)
                layout.addWidget(self.debug, 1)

            def _settings(self) -> None:
                text = self.region_edit.text().strip()
                if text:
                    from blob_solver.vision.region import Region

                    self.controller.select_region(Region.parse(text))
                self.controller.config.board.rows = self.rows.value()
                self.controller.config.board.cols = self.cols.value()
                self.controller.config.solver.mode = self.mode.currentText()
                self.controller.config.solver.time_limit_seconds = self.time_limit.value()
                self.controller.config.solver.beam_width = self.beam_width.value()

            def _start(self, task: Callable[[], object], done: Callable[[object], None] | None = None) -> None:
                if self.worker is not None and self.worker.isRunning():
                    self.status.setText("A task is already running.")
                    return
                self.worker = TaskThread(task)
                self.worker.result.connect(done or self._show_result)
                self.worker.failed.connect(self._show_error)
                self.worker.finished.connect(lambda: setattr(self, "worker", None))
                self.status.setText("Working…")
                self.worker.start()

            def _select_region(self) -> None:
                self._start(
                    lambda: self.controller.select_region(),
                    lambda region: self.region_edit.setText(
                        f"{region.x},{region.y},{region.width},{region.height}"
                    ),
                )

            def _calibrate(self) -> None:
                self._settings()
                self._start(self.controller.calibrate, self._show_observation)

            def _scan(self) -> None:
                self._settings()
                if self.controller.profile is None:
                    self._start(self.controller.calibrate, self._show_observation)
                else:
                    self._start(self.controller.scan, self._show_observation)

            def _analyze(self) -> None:
                self._settings()
                self._start(
                    lambda: self.controller.analyze(
                        mode=self.mode.currentText(),
                        time_limit=self.time_limit.value(),
                    ),
                    self._show_solution,
                )

            def _show_hint(self) -> None:
                self._settings()
                if self.controller.current_solution is not None:
                    spec = self.controller.show_next_move(self.controller.current_solution)
                    self._show_hint_result(spec)
                    return
                self._start(
                    lambda: self.controller.analyze(
                        mode=self.mode.currentText(),
                        time_limit=self.time_limit.value(),
                    ),
                    self._after_analyze_hint,
                )

            def _after_analyze_hint(self, solution: object) -> None:
                spec = self.controller.show_next_move(solution)
                self._show_hint_result(spec)

            def _reread(self) -> None:
                self._start(self.controller.recheck_manual_move, self._show_action)

            def _autoplay(self) -> None:
                self._settings()
                self.controller.config.automation.autoplay = True
                self._start(
                    self.controller.run_autoplay,
                    lambda moves: self.status.setText(
                        f"Autoplay verified {moves} moves; estimated score={self.controller.estimated_score}"
                    ),
                )

            def _stop(self) -> None:
                self.controller.stop_autoplay()
                self.status.setText("Stop requested.")

            def _emergency_stop(self) -> None:
                self.controller.stop_autoplay(emergency=True)
                self.status.setText("EMERGENCY STOP engaged; restart the app/input backend before autoplay.")

            def _show_observation(self, observation: object) -> None:
                if hasattr(observation, "valid") and observation.valid:
                    self.status.setText(
                        f"Vision valid; confidence={observation.min_confidence:.3f}; "
                        f"blocks={observation.board.block_count if observation.board else 0}; "
                        f"legal groups={len(find_groups(observation.board)) if observation.board else 0}"
                    )
                    self.debug.setPlainText(observation.text())
                else:
                    self._show_result(observation)

            def _show_solution(self, solution: object) -> None:
                from blob_solver.cli import format_solution

                self.status.setText(format_solution(solution))
                self.debug.setPlainText(format_solution(solution))

            def _show_hint_result(self, spec: object) -> None:
                self.status.setText("Next move highlighted on the board.")
                if spec is not None:
                    self.debug.appendPlainText(
                        f"\nHighlighted group size={spec.group_size} +{spec.immediate_score} click={spec.click_cell}"
                    )

            def _show_action(self, action: object) -> None:
                self.status.setText("No previous hint." if action is None else f"Readback matched: {action.matched}")
                if action is not None:
                    self.debug.appendPlainText(f"\nActual board:\n{action.observation.text()}")

            def _show_result(self, result: object) -> None:
                self.status.setText(str(result))
                if hasattr(result, "text"):
                    self.debug.setPlainText(result.text())

            def _show_error(self, message: str) -> None:
                self.status.setText(f"Error: {message}")
                QMessageBox.warning(self, "Blob Solver", message)

            def closeEvent(self, event: Any) -> None:
                self.controller.stop_autoplay()
                if self.worker is not None and self.worker.isRunning():
                    self.worker.wait(1000)
                self.controller.desktop.overlay.close()
                event.accept()

            def keyPressEvent(self, event: Any) -> None:
                if event.key() == Qt.Key.Key_Escape:
                    self._emergency_stop()
                    event.accept()
                    return
                super().keyPressEvent(event)

        return Window()
