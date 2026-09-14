"""Small explicit state machine used by the controller/UI."""

from __future__ import annotations

from enum import Enum


class AppState(str, Enum):
    IDLE = "idle"
    REGION_SELECTED = "region_selected"
    CALIBRATING = "calibrating"
    READY = "ready"
    ANALYZING = "analyzing"
    HINT_VISIBLE = "hint_visible"
    WAITING_SETTLE = "waiting_settle"
    AUTOPLAY = "autoplay"
    STOPPED = "stopped"
    ERROR = "error"
