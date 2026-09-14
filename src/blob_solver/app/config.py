"""TOML configuration with safe defaults and last-region persistence."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
import tomllib

from blob_solver.vision.region import Region


@dataclass(slots=True)
class BoardConfig:
    rows: int = 10
    cols: int = 10
    min_group: int = 2
    num_colors: int = 4


@dataclass(slots=True)
class VisionConfig:
    patch_ratio: float = 0.45
    confidence_threshold: float = 0.90
    settle_frames: int = 3
    settle_interval_ms: int = 80
    settle_timeout_seconds: float = 3.0


@dataclass(slots=True)
class SolverConfig:
    mode: str = "hybrid"
    quality: str = "balanced"
    time_limit_seconds: float = 5.0
    beam_width: int = 700
    max_nodes: int = 180_000
    candidate_moves: int = 48
    exact_endgame_blocks: int = 25
    exact_legal_groups: int = 8
    exact_work_limit: int = 140
    rollout_seed: int = 0
    process_count: int = 0


@dataclass(slots=True)
class AutomationConfig:
    click_delay_ms: int = 150
    animation_delay_ms: int = 350
    verified_execution: bool = False


@dataclass(slots=True)
class DesktopConfig:
    backend: str = "auto"
    input_space: str = "logical"


@dataclass(slots=True)
class AppConfig:
    board: BoardConfig = field(default_factory=BoardConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    automation: AutomationConfig = field(default_factory=AutomationConfig)
    desktop: DesktopConfig = field(default_factory=DesktopConfig)
    region: Region | None = None
    calibration: dict[str, object] | None = None
    log_file: str = "blob-solver-session.jsonl"


def _merge_dataclass(instance: object, values: object) -> object:
    if not isinstance(values, dict):
        return instance
    allowed = set(getattr(instance, "__dataclass_fields__", {}))
    updates = {key: value for key, value in values.items() if key in allowed}
    return replace(instance, **updates)


def load_config(path: str | Path | None = None) -> AppConfig:
    config_path = Path(path).expanduser() if path else Path.home() / ".config" / "blob-solver" / "config.toml"
    config = AppConfig()
    if not config_path.exists():
        return config
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)
    config.board = _merge_dataclass(config.board, raw.get("board"))  # type: ignore[assignment]
    config.vision = _merge_dataclass(config.vision, raw.get("vision"))  # type: ignore[assignment]
    config.solver = _merge_dataclass(config.solver, raw.get("solver"))  # type: ignore[assignment]
    config.automation = _merge_dataclass(config.automation, raw.get("automation"))  # type: ignore[assignment]
    config.desktop = _merge_dataclass(config.desktop, raw.get("desktop"))  # type: ignore[assignment]
    region = raw.get("region")
    if isinstance(region, dict):
        config.region = Region.from_dict(region)
    calibration = raw.get("calibration")
    if isinstance(calibration, dict):
        config.calibration = calibration
    if isinstance(raw.get("log_file"), str):
        config.log_file = raw["log_file"]
    return config


def _toml_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return '"' + value.replace('"', '\\"') + '"'
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    return str(value)


def save_config(config: AppConfig, path: str | Path | None = None) -> Path:
    config_path = Path(path).expanduser() if path else Path.home() / ".config" / "blob-solver" / "config.toml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    sections = {
        "board": asdict(config.board),
        "vision": asdict(config.vision),
        "solver": asdict(config.solver),
        "automation": asdict(config.automation),
        "desktop": asdict(config.desktop),
    }
    # Root keys must be emitted before tables; otherwise TOML would attach a
    # later root-looking key to whichever table happened to be open.
    lines: list[str] = [f"log_file = {_toml_value(config.log_file)}", ""]
    for section, values in sections.items():
        lines.append(f"[{section}]")
        lines.extend(f"{key} = {_toml_value(value)}" for key, value in values.items())
        lines.append("")
    if config.region is not None:
        lines.append("[region]")
        lines.extend(f"{key} = {value}" for key, value in config.region.to_dict().items())
        lines.append("")
    if config.calibration:
        lines.append("[calibration]")
        for key, value in config.calibration.items():
            if isinstance(value, dict):
                continue
            lines.append(f"{key} = {_toml_value(value)}")
        colors = config.calibration.get("colors")
        if isinstance(colors, dict):
            lines.append("")
            lines.append("[calibration.colors]")
            for symbol, rgb in colors.items():
                lines.append(f"{symbol} = {_toml_value(rgb)}")
        lines.append("")
    config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return config_path
