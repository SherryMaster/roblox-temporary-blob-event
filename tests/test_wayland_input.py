from __future__ import annotations

import subprocess

import blob_solver.desktop.wayland as wayland
from blob_solver.desktop.wayland import YdotoolInput


def test_ydotool_uses_the_supported_absolute_position_syntax(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run(command, **_kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(wayland.shutil, "which", lambda _command: "/usr/bin/ydotool")
    clicker = YdotoolInput()
    clicker.click(123, 456)

    assert calls[0] == ["ydotool", "mousemove", "--absolute", "-x", "123", "-y", "456"]
    assert calls[1] == ["ydotool", "click", "0xC0"]
