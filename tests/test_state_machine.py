import pytest

from blob_solver.app.state_machine import AppState, InvalidStateTransition, transition


def test_prepared_session_can_rescan_but_cannot_autoplay_without_a_plan() -> None:
    assert transition(AppState.PLAN_READY, AppState.SCANNING) is AppState.SCANNING
    with pytest.raises(InvalidStateTransition):
        transition(AppState.READY_TO_SCAN, AppState.AUTOPLAY)


def test_execution_and_recovery_transitions_are_explicit() -> None:
    assert transition(AppState.PLAN_READY, AppState.SHOWING_STEP) is AppState.SHOWING_STEP
    assert transition(AppState.SHOWING_STEP, AppState.RECOVERY) is AppState.RECOVERY
    assert transition(AppState.RECOVERY, AppState.SCANNING) is AppState.SCANNING
