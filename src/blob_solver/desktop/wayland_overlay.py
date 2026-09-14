"""Native GTK layer-shell overlay service and its JSON-lines client.

The control panel never creates a Wayland overlay widget. This module starts a
small process whose GTK window is initialized with gtk-layer-shell, placed on
the overlay layer, made keyboard-inert, and assigned an empty pointer input
region. The local socket carries only serializable highlight state.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any

from blob_solver.vision.region import Region

from .base import DesktopUnavailable
from .coordinates import CoordinateError, CoordinateMapper, query_hyprland_monitors
from .overlay import HighlightSpec, NullOverlay


_LAYER_SHELL_PROBE = (
    "import gi; "
    "gi.require_version('Gdk', '3.0'); "
    "gi.require_version('Gtk', '3.0'); "
    "gi.require_version('GtkLayerShell', '0.1'); "
    "from gi.repository import Gdk, Gtk, GtkLayerShell"
)


def _service_environment() -> dict[str, str]:
    """Make the source checkout importable by a system GTK interpreter."""

    environment = os.environ.copy()
    source_root = str(Path(__file__).resolve().parents[2])
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = os.pathsep.join(
        part for part in (source_root, existing) if part
    )
    # This process is the Wayland implementation by contract. A desktop-wide
    # GDK_BACKEND=x11 setting must not quietly turn it into an X11 surface.
    environment["GDK_BACKEND"] = "wayland"
    return environment


def _supports_layer_shell(interpreter: str) -> bool:
    try:
        completed = subprocess.run(
            [interpreter, "-c", _LAYER_SHELL_PROBE],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=_service_environment(),
            timeout=1.5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0


def _select_service_python(preferred: str | None) -> str:
    """Use an interpreter that can see gtk-layer-shell, including Arch's system Python.

    Omarchy users often run the control panel from an isolated mise/venv
    interpreter while GTK introspection packages are installed by pacman for
    ``/usr/bin/python``. The overlay is intentionally a separate process, so it
    can use that system interpreter without moving the control panel or solver
    out of the user's environment. An explicit constructor value or
    ``BLOB_SOLVER_WAYLAND_PYTHON`` always wins.
    """

    requested = preferred or os.environ.get("BLOB_SOLVER_WAYLAND_PYTHON")
    if requested:
        return requested
    candidates: list[str] = [sys.executable, "/usr/bin/python3", "/usr/bin/python"]
    seen: set[str] = set()
    for candidate in candidates:
        resolved = shutil.which(candidate) or candidate
        if resolved in seen:
            continue
        seen.add(resolved)
        if _supports_layer_shell(resolved):
            return resolved
    # Preserve a useful service-side dependency error if no interpreter is
    # currently provisioned rather than failing while constructing the client.
    return sys.executable


def _spec_payload(spec: HighlightSpec, mapper: CoordinateMapper) -> dict[str, object]:
    geometry = mapper.overlay_geometry(spec.region)
    return {
        "command": "show",
        "monitor": geometry.monitor.name,
        "monitor_geometry": geometry.monitor.logical_region.to_dict(),
        "region": spec.region.to_dict(),
        "local_region": geometry.local_region.to_dict(),
        "rows": spec.rows,
        "cols": spec.cols,
        "cells": [list(cell) for cell in spec.cells],
        "click_cell": list(spec.click_cell),
        "group_size": spec.group_size,
        "immediate_score": spec.immediate_score,
        "projected_total": spec.projected_total,
        "step": spec.step,
        "step_count": spec.step_count,
        "color": spec.color,
    }


class WaylandLayerShellOverlay(NullOverlay):
    """Client-side handle for the separate native layer-shell process."""

    def __init__(self, *, python: str | None = None, socket_path: str | Path | None = None) -> None:
        super().__init__()
        self._python = _select_service_python(python)
        if socket_path is None:
            directory = tempfile.mkdtemp(prefix="blob-solver-overlay-")
            socket_path = Path(directory) / "overlay.sock"
        self.socket_path = Path(socket_path)
        self._process: subprocess.Popen[bytes] | None = None
        self._connection: socket.socket | None = None
        self._mapper: CoordinateMapper | None = None

    def _ensure_process(self) -> None:
        if self._process is not None and self._process.poll() is None:
            return
        if self._connection is not None:
            self._connection.close()
            self._connection = None
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        self._process = subprocess.Popen(
            [self._python, "-m", "blob_solver.desktop.wayland_overlay", "--service", str(self.socket_path)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            env=_service_environment(),
            close_fds=True,
        )
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                detail = ""
                if self._process.stderr is not None:
                    detail = self._process.stderr.read().decode(errors="replace").strip()
                raise DesktopUnavailable(
                    "Wayland layer-shell service failed to start"
                    + (f": {detail}" if detail else "")
                )
            try:
                self._connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                self._connection.settimeout(1.0)
                self._connection.connect(str(self.socket_path))
                return
            except OSError:
                if self._connection is not None:
                    self._connection.close()
                self._connection = None
                time.sleep(0.03)
        raise DesktopUnavailable("timed out connecting to the Wayland layer-shell service")

    def _send(self, payload: dict[str, object]) -> None:
        self._ensure_process()
        if self._connection is None:
            raise DesktopUnavailable("Wayland layer-shell service is not connected")
        try:
            self._connection.sendall((json.dumps(payload, separators=(",", ":")) + "\n").encode())
            response = b""
            while b"\n" not in response:
                chunk = self._connection.recv(4096)
                if not chunk:
                    raise DesktopUnavailable("Wayland layer-shell service closed its IPC socket")
                response += chunk
            reply = json.loads(response.split(b"\n", 1)[0].decode())
            if not isinstance(reply, dict) or not reply.get("ok", False):
                detail = reply.get("error", "unknown service error") if isinstance(reply, dict) else "invalid service response"
                raise DesktopUnavailable(f"Wayland layer-shell service rejected command: {detail}")
        except (OSError, socket.timeout, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DesktopUnavailable(f"Wayland layer-shell IPC failed: {exc}") from exc

    def show(self, spec: HighlightSpec) -> None:
        try:
            self._mapper = CoordinateMapper(query_hyprland_monitors())
            payload = _spec_payload(spec, self._mapper)
        except CoordinateError as exc:
            raise DesktopUnavailable(str(exc)) from exc
        self.last_spec = spec
        self._send(payload)

    def hide(self) -> None:
        self.last_spec = None
        if self._process is not None and self._process.poll() is None and self._connection is not None:
            self._send({"command": "hide"})

    def close(self) -> None:
        if self._process is not None and self._process.poll() is None and self._connection is not None:
            try:
                self._send({"command": "close"})
            except DesktopUnavailable:
                pass
        if self._connection is not None:
            self._connection.close()
            self._connection = None
        if self._process is not None:
            try:
                self._process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                self._process.terminate()
            self._process = None
        self.last_spec = None


def _require_gi() -> tuple[Any, Any, Any, Any]:
    try:
        import gi

        gi.require_version("Gdk", "3.0")
        gi.require_version("Gtk", "3.0")
        gi.require_version("GtkLayerShell", "0.1")
        from gi.repository import Gdk, Gtk, GtkLayerShell, GLib
    except (ImportError, ValueError) as exc:  # pragma: no cover - desktop dependent
        raise RuntimeError(
            "native Wayland overlay needs python-gobject, GTK3, and gtk-layer-shell "
            "(Arch packages: python-gobject gtk-layer-shell)"
        ) from exc
    return Gdk, Gtk, GtkLayerShell, GLib


def _set_empty_input_region(window: Any, cairo_module: Any) -> None:
    """Set a protocol-level empty pointer region, not just visual opacity."""

    gdk_window = window.get_window()
    if gdk_window is None:
        raise RuntimeError("layer-shell window was not realized")
    empty_region = cairo_module.Region()
    if hasattr(gdk_window, "input_shape_combine_region"):
        # GTK3 exposes the Wayland surface input region through this
        # backend-neutral GDK API. It is not the X11-only visual shape API.
        gdk_window.input_shape_combine_region(empty_region, 0, 0)
    elif hasattr(gdk_window, "set_input_region"):
        gdk_window.set_input_region(empty_region)
    else:  # pragma: no cover - depends on the installed GDK backend
        raise RuntimeError("GTK3 does not expose a surface input-region API")


def _sync_display(gdk_module: Any) -> None:
    """Flush a layer-surface state change before acknowledging IPC.

    The client uses the acknowledgement as the ordering point before a
    capture. GTK normally flushes these requests itself, but an explicit
    display sync closes the small compositor scheduling window in which grim
    could otherwise observe the previous mapped buffer for one frame.
    """

    display = gdk_module.Display.get_default()
    if display is None:
        return
    display.flush()
    if hasattr(display, "sync"):
        display.sync()


def _select_gdk_monitor(display: Any, payload: dict[str, object]) -> Any | None:
    target = payload.get("monitor_geometry")
    if not isinstance(target, dict):
        return None
    target_region = Region.from_dict(target)
    monitors = [display.get_monitor(index) for index in range(display.get_n_monitors())]
    for monitor in monitors:
        geometry = monitor.get_geometry()
        if (
            int(geometry.x) == target_region.x
            and int(geometry.y) == target_region.y
            and int(geometry.width) == target_region.width
            and int(geometry.height) == target_region.height
        ):
            return monitor
    # Some GTK versions expose monitor geometry in physical pixels. Matching
    # the output name is available on newer backends; otherwise the primary
    # monitor is the least surprising fallback and diagnostics remain visible.
    target_name = str(payload.get("monitor", ""))
    for monitor in monitors:
        connector = getattr(monitor, "get_connector", lambda: "")()
        if connector == target_name:
            return monitor
    return display.get_primary_monitor() if hasattr(display, "get_primary_monitor") else None


class _OverlayWindow:
    def __init__(self, Gtk: Any, GtkLayerShell: Any, Gdk: Any, cairo_module: Any, payload: dict[str, object]) -> None:
        self.Gtk = Gtk
        self.GtkLayerShell = GtkLayerShell
        self.Gdk = Gdk
        self.cairo = cairo_module
        self.payload = payload
        self.window = Gtk.Window()
        self.area = Gtk.DrawingArea()
        self.area.connect("draw", self._draw)
        self.window.add(self.area)
        self.window.set_app_paintable(True)
        self.window.set_decorated(False)
        self.window.set_resizable(False)
        GtkLayerShell.init_for_window(self.window)
        GtkLayerShell.set_namespace(self.window, "blob-solver-overlay")
        GtkLayerShell.set_layer(self.window, GtkLayerShell.Layer.OVERLAY)
        if hasattr(GtkLayerShell, "set_keyboard_mode"):
            GtkLayerShell.set_keyboard_mode(self.window, GtkLayerShell.KeyboardMode.NONE)
        else:  # Older gtk-layer-shell releases.
            GtkLayerShell.set_keyboard_interactivity(self.window, False)
        GtkLayerShell.set_exclusive_zone(self.window, 0)
        GtkLayerShell.set_anchor(self.window, GtkLayerShell.Edge.TOP, True)
        GtkLayerShell.set_anchor(self.window, GtkLayerShell.Edge.LEFT, True)
        GtkLayerShell.set_anchor(self.window, GtkLayerShell.Edge.RIGHT, False)
        GtkLayerShell.set_anchor(self.window, GtkLayerShell.Edge.BOTTOM, False)
        self._configure_geometry()
        self.window.connect("realize", self._realized)

    def _configure_geometry(self) -> None:
        local = Region.from_dict(self.payload["local_region"])  # type: ignore[arg-type]
        self.window.set_size_request(local.width, local.height)
        self.GtkLayerShell.set_margin(self.window, self.GtkLayerShell.Edge.TOP, local.y)
        self.GtkLayerShell.set_margin(self.window, self.GtkLayerShell.Edge.LEFT, local.x)
        monitor = _select_gdk_monitor(self.Gdk.Display.get_default(), self.payload)
        if monitor is not None:
            self.GtkLayerShell.set_monitor(self.window, monitor)

    def _realized(self, *_args: object) -> None:
        _set_empty_input_region(self.window, self.cairo)

    def _draw(self, _area: Any, cr: Any) -> bool:
        payload = self.payload
        local = Region.from_dict(payload["local_region"])  # type: ignore[arg-type]
        rows, cols = int(payload["rows"]), int(payload["cols"])
        cells = {tuple(int(item) for item in cell) for cell in payload.get("cells", [])}  # type: ignore[union-attr]
        click = tuple(int(item) for item in payload["click_cell"])  # type: ignore[index]
        for row_from_bottom, col in cells:
            row = rows - 1 - row_from_bottom
            left = round(col * local.width / cols)
            top = round(row * local.height / rows)
            right = round((col + 1) * local.width / cols)
            bottom = round((row + 1) * local.height / rows)
            cr.set_source_rgba(1.0, 0.86, 0.12, 0.16)
            cr.rectangle(left + 3, top + 3, right - left - 6, bottom - top - 6)
            cr.fill_preserve()
            cr.set_source_rgba(1.0, 0.93, 0.35, 0.95)
            cr.set_line_width(2.5)
            cr.stroke()
        if payload.get("color") == "diagnostic":
            # The manual diagnostic deliberately labels every screen cell so
            # monitor offsets, row orientation, and scaling can be checked by
            # eye against the real game board.
            cr.set_font_size(max(8, min(14, min(local.width / cols, local.height / rows) * 0.20)))
            for screen_row in range(rows):
                for col in range(cols):
                    left = round(col * local.width / cols)
                    top = round(screen_row * local.height / rows)
                    right = round((col + 1) * local.width / cols)
                    bottom = round((screen_row + 1) * local.height / rows)
                    cr.set_source_rgba(1.0, 1.0, 1.0, 0.72)
                    cr.set_line_width(1.0)
                    cr.rectangle(left + 1, top + 1, right - left - 2, bottom - top - 2)
                    cr.stroke()
                    cr.move_to(left + 5, top + min(16, max(11, (bottom - top) * 0.28)))
                    cr.show_text(str(screen_row * cols + col + 1))
        row = rows - 1 - click[0]
        left = round(click[1] * local.width / cols)
        top = round(row * local.height / rows)
        right = round((click[1] + 1) * local.width / cols)
        bottom = round((row + 1) * local.height / rows)
        cr.set_source_rgba(1.0, 0.22, 0.18, 0.95)
        cr.set_line_width(3.5)
        cr.arc((left + right) / 2, (top + bottom) / 2, max(7, min(right - left, bottom - top) * 0.31), 0, 6.283)
        cr.stroke()
        cr.set_source_rgba(0.03, 0.04, 0.08, 0.82)
        cr.rectangle(6, 6, 176, 58)
        cr.fill()
        cr.set_source_rgba(1, 1, 1, 0.96)
        cr.set_font_size(13)
        cr.move_to(14, 21)
        cr.show_text(f"{payload['step']} / {payload['step_count']}  {payload['color']}")
        cr.move_to(14, 37)
        cr.show_text(f"+{payload['immediate_score']}  size {payload['group_size']}")
        cr.move_to(14, 53)
        cr.show_text("click target")
        return False

    def show(self) -> None:
        self.window.show_all()
        _set_empty_input_region(self.window, self.cairo)
        self.area.queue_draw()

    def hide(self) -> None:
        self.window.hide()

    def close(self) -> None:
        self.window.destroy()


def run_overlay_service(socket_path: str | Path) -> int:  # pragma: no cover - requires live Wayland
    os.environ.setdefault("GDK_BACKEND", "wayland")
    Gdk, Gtk, GtkLayerShell, GLib = _require_gi()
    Gtk.init([])
    try:
        import cairo
    except ImportError as exc:
        raise RuntimeError("pycairo is required by the GTK layer-shell overlay") from exc
    socket_file = Path(socket_path)
    socket_file.unlink(missing_ok=True)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(socket_file))
    server.listen(1)
    server.settimeout(0.5)
    current: _OverlayWindow | None = None
    stop = False

    def handle(payload: dict[str, object], connection: socket.socket) -> bool:
        nonlocal current, stop
        try:
            command = payload.get("command")
            if command == "show":
                if current is not None:
                    current.close()
                current = _OverlayWindow(Gtk, GtkLayerShell, Gdk, cairo, payload)
                current.show()
                _sync_display(Gdk)
            elif command == "hide" and current is not None:
                current.hide()
                _sync_display(Gdk)
            elif command == "close":
                if current is not None:
                    current.close()
                    current = None
                _sync_display(Gdk)
                stop = True
            else:
                raise ValueError(f"unknown overlay command: {command!r}")
            connection.sendall(b'{"ok":true}\n')
        except Exception as exc:
            try:
                connection.sendall((json.dumps({"ok": False, "error": str(exc)}) + "\n").encode())
            except OSError:
                pass
            return False
        if payload.get("command") == "close":
            Gtk.main_quit()
        return False

    def reader() -> None:
        nonlocal stop
        while not stop:
            try:
                connection, _ = server.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            with connection:
                buffer = b""
                while not stop:
                    chunk = connection.recv(65536)
                    if not chunk:
                        break
                    buffer += chunk
                    while b"\n" in buffer:
                        line, buffer = buffer.split(b"\n", 1)
                        if not line:
                            continue
                        try:
                            payload = json.loads(line.decode())
                        except (UnicodeDecodeError, json.JSONDecodeError):
                            continue
                        if isinstance(payload, dict):
                            GLib.idle_add(handle, payload, connection)

    import threading

    threading.Thread(target=reader, name="blob-overlay-ipc", daemon=True).start()
    Gtk.main()
    stop = True
    server.close()
    socket_file.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--service":
        try:
            exit_code = run_overlay_service(sys.argv[2])
        except Exception as exc:
            print(f"blob-solver Wayland overlay service: {exc}", file=sys.stderr)
            exit_code = 1
        raise SystemExit(exit_code)
    raise SystemExit("usage: python -m blob_solver.desktop.wayland_overlay --service SOCKET")
