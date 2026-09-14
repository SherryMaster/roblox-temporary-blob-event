"""Small explicit state machine used by the controller/UI."""

from __future__ import annotations

from enum import Enum


class AppState(str, Enum):
    NO_REGION = "no_region"
    READY_TO_SCAN = "ready_to_scan"
    SCANNING = "scanning"
    SEARCHING = "searching"
    PLAN_READY = "plan_ready"
    SHOWING_STEP = "showing_step"
    AUTOPLAY = "autoplay"
    PAUSED = "paused"
    RECOVERY = "recovery"
    ERROR = "error"

    # Compatibility names for callers written against the initial milestone.
    IDLE = "no_region"
    REGION_SELECTED = "ready_to_scan"
    CALIBRATING = "scanning"
    READY = "ready_to_scan"
    ANALYZING = "searching"
    HINT_VISIBLE = "showing_step"
    WAITING_SETTLE = "recovery"
    STOPPED = "paused"


class InvalidStateTransition(RuntimeError):
    pass


_ALLOWED: dict[AppState, frozenset[AppState]] = {
    AppState.NO_REGION: frozenset({AppState.READY_TO_SCAN, AppState.SCANNING, AppState.ERROR}),
    AppState.READY_TO_SCAN: frozenset({AppState.NO_REGION, AppState.SCANNING, AppState.SEARCHING, AppState.ERROR}),
    AppState.SCANNING: frozenset({AppState.READY_TO_SCAN, AppState.SEARCHING, AppState.PAUSED, AppState.ERROR, AppState.RECOVERY}),
    AppState.SEARCHING: frozenset({AppState.PLAN_READY, AppState.READY_TO_SCAN, AppState.ERROR, AppState.PAUSED}),
    AppState.PLAN_READY: frozenset({AppState.SHOWING_STEP, AppState.AUTOPLAY, AppState.RECOVERY, AppState.PAUSED, AppState.SCANNING, AppState.ERROR}),
    AppState.SHOWING_STEP: frozenset({AppState.SHOWING_STEP, AppState.PLAN_READY, AppState.AUTOPLAY, AppState.PAUSED, AppState.RECOVERY, AppState.ERROR}),
    AppState.AUTOPLAY: frozenset({AppState.SHOWING_STEP, AppState.SEARCHING, AppState.PAUSED, AppState.PLAN_READY, AppState.RECOVERY, AppState.ERROR}),
    AppState.PAUSED: frozenset({AppState.PLAN_READY, AppState.SHOWING_STEP, AppState.AUTOPLAY, AppState.RECOVERY, AppState.READY_TO_SCAN, AppState.SCANNING, AppState.ERROR}),
    AppState.RECOVERY: frozenset({AppState.SCANNING, AppState.SEARCHING, AppState.PLAN_READY, AppState.READY_TO_SCAN, AppState.ERROR}),
    AppState.ERROR: frozenset({AppState.NO_REGION, AppState.READY_TO_SCAN, AppState.SCANNING, AppState.RECOVERY}),
}


def transition(current: AppState, target: AppState) -> AppState:
    """Validate a user-visible lifecycle transition."""

    if target == current:
        return target
    if target not in _ALLOWED.get(current, frozenset()):
        raise InvalidStateTransition(f"cannot transition from {current.value} to {target.value}")
    return target
