"""Best-effort transparent Qt overlay with an explicit preview fallback."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

from blob_solver.vision.region import Region


@dataclass(frozen=True, slots=True)
class HighlightSpec:
    region: Region
    rows: int
    cols: int
    cells: tuple[tuple[int, int], ...]
    click_cell: tuple[int, int]
    group_size: int
    immediate_score: int
    projected_total: int


class NullOverlay:
    """No-op overlay used by CLI/tests and as a safe Wayland fallback."""

    def __init__(self) -> None:
        self.last_spec: HighlightSpec | None = None

    def show(self, spec: HighlightSpec) -> None:
        self.last_spec = spec

    def hide(self) -> None:
        self.last_spec = None

    def close(self) -> None:
        self.hide()

    @contextmanager
    def hidden_during_capture(self) -> Iterator[None]:
        previous = self.last_spec
        self.hide()
        try:
            yield
        finally:
            if previous is not None:
                self.show(previous)


class QtOverlay(NullOverlay):
    """Transparent, mouse-pass-through overlay when PySide6 supports it."""

    def __init__(self, parent: Any = None) -> None:
        try:
            from PySide6.QtCore import Qt
            from PySide6.QtGui import QColor, QPainter, QPen
            from PySide6.QtWidgets import QWidget
        except ImportError as exc:  # pragma: no cover - optional desktop UI
            raise RuntimeError("PySide6 is required for the transparent overlay") from exc
        self._Qt = Qt
        self._QColor = QColor
        self._QPainter = QPainter
        self._QPen = QPen
        self._widget_class = QWidget
        self._widget = _make_overlay_widget(self, parent)
        self.last_spec: HighlightSpec | None = None

    def show(self, spec: HighlightSpec) -> None:
        self.last_spec = spec
        self._widget.setGeometry(spec.region.x, spec.region.y, spec.region.width, spec.region.height)
        self._widget.update()
        self._widget.show()
        self._widget.raise_()

    def hide(self) -> None:
        self.last_spec = None
        self._widget.hide()

    def close(self) -> None:
        self._widget.close()


def _make_overlay_widget(overlay: QtOverlay, parent: Any = None) -> Any:
    QWidget = overlay._widget_class
    Qt = overlay._Qt

    class Widget(QWidget):
        def __init__(self) -> None:
            super().__init__(parent)
            self.overlay = overlay
            self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
            self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
            self.setWindowFlags(
                Qt.WindowType.Tool
                | Qt.WindowType.FramelessWindowHint
                | Qt.WindowType.WindowStaysOnTopHint
                | Qt.WindowType.BypassWindowManagerHint
            )

        def paintEvent(self, event: Any) -> None:
            spec = self.overlay.last_spec
            if spec is None:
                return
            painter = self.overlay._QPainter(self)
            painter.setRenderHint(self.overlay._QPainter.RenderHint.Antialiasing)
            painter.setPen(self.overlay._QPen(self.overlay._QColor(255, 240, 80, 255), 4))
            for row_from_bottom, col in spec.cells:
                row = spec.rows - 1 - row_from_bottom
                left, top, right, bottom = spec.region.cell_rect(spec.rows, spec.cols, row, col)
                painter.drawRoundedRect(left + 3, top + 3, right - left - 6, bottom - top - 6, 8, 8)
            row = spec.rows - 1 - spec.click_cell[0]
            left, top, right, bottom = spec.region.cell_rect(spec.rows, spec.cols, row, spec.click_cell[1])
            painter.setPen(self.overlay._QPen(self.overlay._QColor(255, 80, 80, 255), 5))
            painter.drawEllipse(left + 8, top + 8, right - left - 16, bottom - top - 16)
            painter.drawText(
                8,
                24,
                f"CLICK  Group: {spec.group_size}  +{spec.immediate_score}  Projected: {spec.projected_total}",
            )
            painter.end()

    return Widget()


def create_overlay(*, prefer_qt: bool = True) -> NullOverlay:
    if prefer_qt:
        try:
            from PySide6.QtWidgets import QApplication

            if QApplication.instance() is None:
                return NullOverlay()
            return QtOverlay()
        except (ImportError, RuntimeError):
            pass
    return NullOverlay()
