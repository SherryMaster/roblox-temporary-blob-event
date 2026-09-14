"""Compact PySide6 control panel for single-scan planning and frozen execution."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .config import load_config
from .controller import AutomationController


def run_app(*, config_path: str | Path | None = None) -> int:
    try:
        from PySide6.QtCore import QThread, Qt, Signal
        from PySide6.QtGui import QColor, QPainter, QPen
        from PySide6.QtWidgets import (
            QApplication,
            QComboBox,
            QCheckBox,
            QFormLayout,
            QGroupBox,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QListWidget,
            QListWidgetItem,
            QMainWindow,
            QMessageBox,
            QPushButton,
            QSpinBox,
            QDoubleSpinBox,
            QVBoxLayout,
            QWidget,
        )
    except ImportError as exc:  # pragma: no cover - optional UI
        raise RuntimeError("install the ui extra with: pip install 'blob-solver[ui]'") from exc

    from blob_solver.game.board import Board
    from blob_solver.vision.region import Region

    palette = {
        "red": QColor(238, 83, 91),
        "yellow": QColor(246, 194, 57),
        "green": QColor(126, 188, 48),
        "blue": QColor(38, 130, 193),
        "R": QColor(238, 83, 91),
        "Y": QColor(246, 194, 57),
        "G": QColor(126, 188, 48),
        "B": QColor(38, 130, 193),
    }

    class BoardWidget(QWidget):
        """Small graphical reconstruction/timeline board, not a text dump."""

        def __init__(self, parent: QWidget | None = None) -> None:
            super().__init__(parent)
            self.board: Board | None = None
            self.rows = 10
            self.cols = 10
            self.selected: set[tuple[int, int]] = set()
            self.click_cell: tuple[int, int] | None = None
            self.setMinimumSize(360, 360)

        def set_state(
            self,
            board: Board | None,
            *,
            rows: int,
            cols: int,
            selected: tuple[tuple[int, int], ...] = (),
            click_cell: tuple[int, int] | None = None,
        ) -> None:
            self.board = board
            self.rows = rows
            self.cols = cols
            self.selected = set(selected)
            self.click_cell = click_cell
            self.update()

        def paintEvent(self, _event: Any) -> None:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.fillRect(self.rect(), QColor(14, 20, 39))
            margin = 12
            cell_width = (self.width() - 2 * margin) / self.cols
            cell_height = (self.height() - 2 * margin) / self.rows
            matrix = (
                self.board.to_matrix(rows=self.rows, cols=self.cols, empty=None)
                if self.board is not None
                else tuple(tuple(None for _ in range(self.cols)) for _ in range(self.rows))
            )
            for row in range(self.rows):
                for col in range(self.cols):
                    left = round(margin + col * cell_width)
                    top = round(margin + row * cell_height)
                    right = round(margin + (col + 1) * cell_width)
                    bottom = round(margin + (row + 1) * cell_height)
                    value = matrix[row][col]
                    base = palette.get(str(value), QColor(24, 34, 60)) if value is not None else QColor(24, 34, 60)
                    painter.setPen(QPen(QColor(46, 59, 88), 1))
                    painter.setBrush(base)
                    painter.drawRoundedRect(left + 2, top + 2, right - left - 4, bottom - top - 4, 7, 7)
                    if value is not None:
                        painter.setPen(QPen(QColor(255, 255, 255, 55), 1))
                        painter.drawRoundedRect(left + 4, top + 4, right - left - 8, bottom - top - 8, 5, 5)
                    logical = (self.rows - 1 - row, col)
                    if logical in self.selected:
                        painter.setBrush(QColor(255, 221, 63, 45))
                        painter.setPen(QPen(QColor(255, 235, 102), 3))
                        painter.drawRoundedRect(left + 3, top + 3, right - left - 6, bottom - top - 6, 7, 7)
                    if logical == self.click_cell:
                        painter.setBrush(Qt.BrushStyle.NoBrush)
                        painter.setPen(QPen(QColor(255, 86, 74), 3))
                        painter.drawEllipse(left + 9, top + 9, right - left - 18, bottom - top - 18)
            painter.end()

    class TaskThread(QThread):
        result = Signal(object)
        failed = Signal(str)
        progress = Signal(object)

        def __init__(self, task: Callable[[Callable[[object], None]], object]) -> None:
            super().__init__()
            self.task = task

        def run(self) -> None:
            try:
                self.result.emit(self.task(self.progress.emit))
            except Exception as exc:
                self.failed.emit(str(exc))

    class Window(QMainWindow):
        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle("Blob Solver  •  deterministic planner")
            self.resize(1080, 760)
            self.config_path = config_path
            self.controller = AutomationController(load_config(config_path), config_path=config_path)
            self.worker: TaskThread | None = None
            self._build()

        def _build(self) -> None:
            root = QWidget()
            self.setCentralWidget(root)
            outer = QVBoxLayout(root)
            outer.setContentsMargins(18, 18, 18, 18)
            outer.setSpacing(12)

            title_row = QHBoxLayout()
            title = QLabel("BLOB SOLVER")
            title.setObjectName("title")
            subtitle = QLabel("One capture  •  complete local plan  •  frozen execution")
            subtitle.setObjectName("subtitle")
            title_row.addWidget(title)
            title_row.addWidget(subtitle)
            title_row.addStretch(1)
            outer.addLayout(title_row)

            setup = QGroupBox("SETUP")
            setup_layout = QFormLayout(setup)
            region_row = QHBoxLayout()
            self.region_edit = QLineEdit()
            if self.controller.region:
                region = self.controller.region
                self.region_edit.setText(f"{region.x},{region.y},{region.width},{region.height}")
            select = QPushButton("Select Region")
            select.clicked.connect(self._select_region)
            region_row.addWidget(self.region_edit, 1)
            region_row.addWidget(select)
            setup_layout.addRow("Board region", region_row)
            grid_row = QHBoxLayout()
            self.rows = QSpinBox()
            self.rows.setRange(1, 100)
            self.rows.setValue(self.controller.config.board.rows)
            self.cols = QSpinBox()
            self.cols.setRange(1, 100)
            self.cols.setValue(self.controller.config.board.cols)
            grid_row.addWidget(QLabel("Rows"))
            grid_row.addWidget(self.rows)
            grid_row.addWidget(QLabel("Columns"))
            grid_row.addWidget(self.cols)
            grid_row.addStretch(1)
            setup_layout.addRow("Grid", grid_row)
            scan_row = QHBoxLayout()
            self.quality = QComboBox()
            self.quality.addItems(["fast", "balanced", "deep", "exhaustive"])
            self.quality.setCurrentText(self.controller.config.solver.quality)
            scan = QPushButton("SCAN & SOLVE")
            scan.setObjectName("primary")
            scan.clicked.connect(self._scan_and_solve)
            scan_row.addWidget(self.quality)
            scan_row.addWidget(scan)
            setup_layout.addRow("Quality", scan_row)
            outer.addWidget(setup)

            plan_box = QGroupBox("PLAN")
            plan_layout = QHBoxLayout(plan_box)
            board_column = QVBoxLayout()
            self.board_widget = BoardWidget()
            board_column.addWidget(self.board_widget, 1)
            self.plan_summary = QLabel("No frozen generation yet.")
            self.plan_summary.setWordWrap(True)
            board_column.addWidget(self.plan_summary)
            plan_layout.addLayout(board_column, 2)
            timeline_column = QVBoxLayout()
            timeline_column.addWidget(QLabel("Complete sequence"))
            self.timeline = QListWidget()
            self.timeline.currentRowChanged.connect(self._inspect_step)
            timeline_column.addWidget(self.timeline, 1)
            plan_layout.addLayout(timeline_column, 1)
            outer.addWidget(plan_box, 1)

            execution = QGroupBox("EXECUTION")
            execution_layout = QHBoxLayout(execution)
            self.show_button = QPushButton("Show Step")
            self.previous_button = QPushButton("Previous Step")
            self.next_button = QPushButton("Next Step")
            self.autoplay_button = QPushButton("Auto Play")
            self.pause_button = QPushButton("Pause / Stop Search")
            recover = QPushButton("RESCAN / RECOVER")
            recover.setObjectName("warning")
            self.show_button.clicked.connect(self._show_step)
            self.previous_button.clicked.connect(self._previous)
            self.next_button.clicked.connect(self._next)
            self.autoplay_button.clicked.connect(self._autoplay)
            self.pause_button.clicked.connect(self._pause)
            recover.clicked.connect(self._recover)
            for button in (self.show_button, self.previous_button, self.next_button, self.autoplay_button, self.pause_button, recover):
                execution_layout.addWidget(button)
            outer.addWidget(execution)

            advanced = QGroupBox("Advanced")
            advanced.setCheckable(True)
            advanced.setChecked(False)
            advanced_layout = QFormLayout(advanced)
            self.time_limit = QDoubleSpinBox()
            self.time_limit.setRange(0.0, 3600.0)
            self.time_limit.setDecimals(2)
            self.time_limit.setSpecialValueText("quality preset")
            self.time_limit.setValue(0.0)
            self.beam_width = QSpinBox()
            self.beam_width.setRange(1, 1_000_000)
            self.beam_width.setValue(self.controller.config.solver.beam_width)
            self.process_count = QSpinBox()
            self.process_count.setRange(0, 128)
            self.process_count.setSpecialValueText("automatic")
            self.process_count.setValue(self.controller.config.solver.process_count)
            self.animation_delay = QSpinBox()
            self.animation_delay.setRange(0, 10_000)
            self.animation_delay.setSuffix(" ms")
            self.animation_delay.setValue(self.controller.config.automation.animation_delay_ms)
            self.verified_execution = QCheckBox("capture after every move")
            self.verified_execution.setChecked(self.controller.config.automation.verified_execution)
            advanced_layout.addRow("Search time override (s)", self.time_limit)
            advanced_layout.addRow("Beam capacity", self.beam_width)
            advanced_layout.addRow("Root worker processes", self.process_count)
            advanced_layout.addRow("Animation delay", self.animation_delay)
            advanced_layout.addRow("Verified execution", self.verified_execution)
            outer.addWidget(advanced)

            self.status = QLabel("Ready to scan.")
            self.status.setObjectName("status")
            self.status.setWordWrap(True)
            outer.addWidget(self.status)
            self.setStyleSheet(
                """
                QWidget { background: #0c1224; color: #e8edf8; font-size: 13px; }
                QGroupBox { border: 1px solid #263452; border-radius: 9px; margin-top: 10px; padding: 12px; }
                QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 5px; color: #94a8d3; }
                QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QListWidget { background: #141e38; border: 1px solid #2c3c62; border-radius: 6px; padding: 6px; }
                QPushButton { background: #1c2a4c; border: 1px solid #3e5583; border-radius: 6px; padding: 8px 12px; }
                QPushButton:hover { background: #29406e; }
                QPushButton#primary { background: #2f76d2; border-color: #6ea7ff; font-weight: 700; }
                QPushButton#warning { background: #71342f; border-color: #cf7469; }
                QLabel#title { color: #8fc4ff; font-size: 20px; font-weight: 800; letter-spacing: 2px; }
                QLabel#subtitle { color: #8998b8; padding-left: 10px; }
                QLabel#status { background: #121d36; border-radius: 6px; padding: 8px; color: #b8c8e9; }
                QListWidget::item:selected { background: #2b4b7d; border-radius: 4px; }
                """
            )

        def _settings(self) -> None:
            text = self.region_edit.text().strip()
            if text:
                region = Region.parse(text)
                if region != self.controller.region:
                    self.controller.select_region(region)
            self.controller.config.board.rows = self.rows.value()
            self.controller.config.board.cols = self.cols.value()
            self.controller.config.solver.quality = self.quality.currentText()
            self.controller.config.solver.time_limit_seconds = self.time_limit.value()
            self.controller.config.solver.beam_width = self.beam_width.value()
            self.controller.config.solver.process_count = self.process_count.value()
            self.controller.config.automation.animation_delay_ms = self.animation_delay.value()
            self.controller.config.automation.verified_execution = self.verified_execution.isChecked()

        def _selected_time_limit(self) -> float | None:
            value = self.time_limit.value()
            return None if value <= 0 else value

        def _start(self, task: Callable[[Callable[[object], None]], object], done: Callable[[object], None]) -> None:
            if self.worker is not None and self.worker.isRunning():
                self.status.setText("A scan or search is already running.")
                return
            self.worker = TaskThread(task)
            self.worker.result.connect(done)
            self.worker.failed.connect(self._show_error)
            self.worker.progress.connect(self._show_progress)
            self.worker.finished.connect(self._worker_finished)
            self.status.setText("Working…")
            self.worker.start()

        def _worker_finished(self) -> None:
            self.worker = None

        def _select_region(self) -> None:
            self._start(lambda _progress: self.controller.select_region(), self._region_done)

        def _region_done(self, region: object) -> None:
            self.region_edit.setText(f"{region.x},{region.y},{region.width},{region.height}")
            self.status.setText("Region selected. Press SCAN & SOLVE.")

        def _scan_and_solve(self) -> None:
            self._settings()
            self._start(
                lambda progress: self.controller.scan_and_solve(
                    quality=self.quality.currentText(),
                    time_limit=self._selected_time_limit(),
                    progress_callback=progress,
                ),
                self._plan_done,
            )

        def _plan_done(self, plan: object) -> None:
            self._populate_plan(plan)
            self.status.setText(
                f"{plan.status}  •  projected {plan.total_score}  •  {plan.move_count} moves  •  "
                f"{plan.leftover_blocks} leftover blocks"
            )

        def _populate_plan(self, plan: object) -> None:
            self.timeline.blockSignals(True)
            self.timeline.clear()
            for step in plan.steps:
                item = QListWidgetItem(
                    f"Step {step.index:2d}   {step.move.color} ×{step.group_size}   +{step.immediate_score}   → {step.cumulative_score}"
                )
                self.timeline.addItem(item)
            self.timeline.blockSignals(False)
            stats = dict(plan.solver_stats)
            search_time = float(stats.get("search_time_seconds", 0.0))
            states = int(stats.get("states_examined", stats.get("nodes", 0)))
            terminals = int(stats.get("terminal_plans", 0))
            self.plan_summary.setText(
                f"{plan.status}\nProjected total: {plan.total_score}\n"
                f"Cleared: {plan.blocks_cleared}/{plan.initial_board.block_count}  •  "
                f"Leftover: {plan.leftover_blocks}\nGap ≤ {plan.gap}  •  "
                f"Upper bound: {plan.upper_bound}\n"
                f"Optimal proven: {'yes' if plan.optimal_proven else 'no'}\n"
                f"Search: {search_time:.2f}s  •  States: {states:,}  •  Terminal plans: {terminals:,}"
            )
            self._inspect_step(0 if plan.steps else -1)

        def _show_progress(self, progress: object) -> None:
            # The first hybrid incumbent is emitted after the one captured
            # image has already been classified. Paint that frozen
            # reconstruction immediately while deeper search continues.
            observation = self.controller.current_observation
            if observation is not None and self.controller.active_plan is None:
                self.board_widget.set_state(
                    observation.board,
                    rows=self.controller.config.board.rows,
                    cols=self.controller.config.board.cols,
                )
            self.status.setText(
                f"Searching…  Best complete score: {progress.best_complete_score}  •  "
                f"Upper bound: {progress.upper_bound}  •  States: {progress.states_examined}  •  "
                f"Terminal plans: {progress.terminal_plans}  •  {progress.elapsed_seconds:.1f}s"
            )

        def _inspect_step(self, index: int) -> None:
            plan = self.controller.active_plan
            if plan is None or index < 0 or index >= len(plan.steps):
                return
            step = self.controller.session.view(index)
            if step is None:
                return
            self.board_widget.set_state(
                step.before_board,
                rows=self.controller.config.board.rows,
                cols=self.controller.config.board.cols,
                selected=step.move.cells,
                click_cell=step.click_cell,
            )
            self.status.setText(
                f"Step {step.index} / {plan.move_count}  •  Click highlighted {step.move.color} ×{step.group_size}  •  "
                f"+{step.immediate_score}  •  projected final {plan.total_score}"
            )

        def _show_step(self) -> None:
            try:
                self.controller.show_step()
                self._inspect_step(self.controller.current_step)
            except Exception as exc:
                self._show_error(str(exc))

        def _previous(self) -> None:
            try:
                step = self.controller.previous_step()
                if step is not None:
                    self.timeline.setCurrentRow(step.index - 1)
            except Exception as exc:
                self._show_error(str(exc))

        def _next(self) -> None:
            try:
                step = self.controller.next_step()
                if step is not None:
                    self.timeline.setCurrentRow(step.index - 1)
                    self.controller.show_step()
                else:
                    self.status.setText("Frozen plan complete.")
            except Exception as exc:
                self._show_error(str(exc))

        def _autoplay(self) -> None:
            self._settings()
            self._start(lambda _progress: self.controller.run_autoplay(), self._autoplay_done)

        def _autoplay_done(self, moves: object) -> None:
            self.status.setText(f"Auto Play finished: {moves} frozen steps executed; no normal rescan was used.")

        def _pause(self) -> None:
            self.controller.stop_autoplay()
            self.status.setText("Pause/stop requested; the frozen plan is still available.")

        def _recover(self) -> None:
            self._settings()
            self._start(
                lambda progress: self.controller.rescan_recover(
                    quality=self.quality.currentText(),
                    time_limit=self._selected_time_limit(),
                    progress_callback=progress,
                ),
                self._plan_done,
            )

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
                self.controller.stop_autoplay(emergency=True)
                self.status.setText("Emergency stop engaged.")
                event.accept()
                return
            super().keyPressEvent(event)

    application = QApplication.instance() or QApplication([])
    try:
        window = Window()
    except Exception as exc:
        QMessageBox.critical(None, "Blob Solver", str(exc))
        return 2
    window.show()
    return application.exec()
