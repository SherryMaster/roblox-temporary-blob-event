from blob_solver.desktop.coordinates import CoordinateMapper, monitor_info_from_json
from blob_solver.desktop.overlay import HighlightSpec
from blob_solver.desktop import wayland_overlay
from blob_solver.desktop.wayland_overlay import _spec_payload
from blob_solver.vision.region import Region


def test_wayland_service_can_select_an_arch_python_with_gi(monkeypatch) -> None:
    monkeypatch.setattr(wayland_overlay.sys, "executable", "/isolated/python")
    monkeypatch.setattr(
        wayland_overlay,
        "_supports_layer_shell",
        lambda interpreter: interpreter == "/usr/bin/python3",
    )

    assert wayland_overlay._select_service_python(None) == "/usr/bin/python3"


def test_hyprland_scale_and_local_overlay_coordinates_are_explicit() -> None:
    monitor = monitor_info_from_json(
        {
            "name": "DP-1",
            "x": 0,
            "y": 0,
            "width": 2560,
            "height": 1440,
            "scale": 2.0,
            "focused": True,
            "transform": 0,
        }
    )
    mapper = CoordinateMapper((monitor,), input_space="physical")
    region = Region(100, 50, 200, 100)

    geometry = mapper.overlay_geometry(region)
    diagnostic = mapper.click_diagnostic(
        region,
        rows=10,
        cols=10,
        screen_row=2,
        col=3,
    )

    assert monitor.logical_region == Region(0, 0, 1280, 720)
    assert geometry.local_region == region
    assert diagnostic.logical_click_point == (170, 75)
    assert diagnostic.input_click_point == (340, 150)
    assert diagnostic.monitor == "DP-1"


def test_monitor_offsets_are_removed_only_for_layer_local_geometry() -> None:
    monitor = monitor_info_from_json(
        {
            "name": "HDMI-A-1",
            "x": -1280,
            "y": 40,
            "width": 2560,
            "height": 1440,
            "scale": 2.0,
        }
    )
    mapper = CoordinateMapper((monitor,), input_space="physical")
    region = Region(-1180, 100, 200, 100)

    geometry = mapper.overlay_geometry(region)
    diagnostic = mapper.click_diagnostic(
        region,
        rows=10,
        cols=10,
        screen_row=2,
        col=3,
    )

    assert geometry.local_region == Region(100, 60, 200, 100)
    assert diagnostic.logical_click_point == (-1110, 125)
    assert diagnostic.input_click_point == (-2220, 250)


def test_wayland_payload_contains_the_whole_group_and_step_label() -> None:
    monitor = monitor_info_from_json(
        {
            "name": "eDP-1",
            "x": 0,
            "y": 0,
            "width": 1920,
            "height": 1080,
            "scale": 1,
        }
    )
    spec = HighlightSpec(
        region=Region(20, 30, 100, 100),
        rows=10,
        cols=10,
        cells=((0, 1), (1, 1), (2, 1)),
        click_cell=(1, 1),
        group_size=3,
        immediate_score=9,
        projected_total=123,
        step=4,
        step_count=22,
        color="green",
    )

    payload = _spec_payload(spec, CoordinateMapper((monitor,)))

    assert payload["cells"] == [[0, 1], [1, 1], [2, 1]]
    assert payload["click_cell"] == [1, 1]
    assert payload["step"] == 4
    assert payload["step_count"] == 22
    assert payload["color"] == "green"
