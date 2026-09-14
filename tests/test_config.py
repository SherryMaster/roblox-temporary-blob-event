from blob_solver.app.config import AppConfig, load_config, save_config
from blob_solver.vision.calibration import CalibrationProfile
from blob_solver.vision.region import Region


def test_config_round_trip_preserves_region_and_calibration(tmp_path) -> None:
    config = AppConfig(
        region=Region(1, 2, 300, 400),
        calibration=CalibrationProfile(
            colors=(("red", (250, 80, 90)), ("blue", (20, 120, 190))),
        ).to_dict(),
    )
    path = tmp_path / "config.toml"
    save_config(config, path)
    loaded = load_config(path)
    assert loaded.region == config.region
    assert loaded.calibration == config.calibration
    assert loaded.log_file == config.log_file
