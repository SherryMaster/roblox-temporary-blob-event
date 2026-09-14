"""Dynamic color calibration without making screenshot RGB values constants."""

from __future__ import annotations

from dataclasses import dataclass
import colorsys
import math
from typing import Iterable, Sequence

from .grid import CellSample

RGB = tuple[int, int, int]

DEFAULT_SYMBOLS = ("red", "yellow", "green", "blue")
DEFAULT_PALETTE: dict[str, RGB] = {
    "yellow": (251, 199, 57),
    "red": (251, 88, 93),
    "green": (136, 198, 37),
    "blue": (25, 128, 193),
}


def _distance(a: RGB, b: RGB) -> float:
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def _brightness(rgb: RGB) -> float:
    return max(rgb) / 255.0


def _saturation(rgb: RGB) -> float:
    return colorsys.rgb_to_hsv(*(channel / 255.0 for channel in rgb))[1]


def likely_background(rgb: RGB) -> bool:
    """Conservative navy/dark-background detector used only during calibration."""

    return _brightness(rgb) < 0.42 and _saturation(rgb) < 0.9


def _mean(points: Sequence[RGB]) -> RGB:
    return tuple(int(round(sum(point[index] for point in points) / len(points))) for index in range(3))  # type: ignore[return-value]


def _cluster(points: Sequence[RGB], cluster_count: int, iterations: int = 24) -> list[RGB]:
    """Small deterministic k-means implementation for at most a few colors."""

    unique = list(dict.fromkeys(points))
    if not unique:
        return []
    cluster_count = min(cluster_count, len(unique))
    # Farthest-first initialization is stable and avoids requiring random state.
    centroids = [unique[0]]
    while len(centroids) < cluster_count:
        candidate = max(unique, key=lambda point: min(_distance(point, center) for center in centroids))
        centroids.append(candidate)
    for _ in range(iterations):
        buckets: list[list[RGB]] = [[] for _ in centroids]
        for point in points:
            index = min(range(len(centroids)), key=lambda item: _distance(point, centroids[item]))
            buckets[index].append(point)
        next_centroids = [
            _mean(bucket) if bucket else centroids[index]
            for index, bucket in enumerate(buckets)
        ]
        if next_centroids == centroids:
            break
        centroids = next_centroids
    return centroids


def _match_symbols(centroids: Sequence[RGB], symbols: Sequence[str]) -> tuple[tuple[str, RGB], ...]:
    available = list(symbols)
    pairs: list[tuple[str, RGB]] = []
    for centroid in centroids:
        if available:
            symbol = min(available, key=lambda item: _distance(centroid, DEFAULT_PALETTE.get(item, centroid)))
            available.remove(symbol)
        else:
            symbol = f"color{len(pairs)}"
        pairs.append((symbol, centroid))
    return tuple(pairs)


@dataclass(frozen=True, slots=True)
class CalibrationProfile:
    colors: tuple[tuple[str, RGB], ...]
    background: RGB = (16, 29, 61)
    empty_distance_factor: float = 0.58
    unknown_distance: float = 115.0

    def __post_init__(self) -> None:
        if not self.colors:
            raise ValueError("calibration needs at least one playable color")

    @property
    def centroids(self) -> dict[str, RGB]:
        return dict(self.colors)

    def to_dict(self) -> dict[str, object]:
        return {
            "colors": {symbol: list(rgb) for symbol, rgb in self.colors},
            "background": list(self.background),
            "empty_distance_factor": self.empty_distance_factor,
            "unknown_distance": self.unknown_distance,
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "CalibrationProfile":
        raw_colors = value.get("colors", {})
        if not isinstance(raw_colors, dict):
            raise ValueError("calibration colors must be a table")
        colors = tuple((str(symbol), tuple(int(channel) for channel in rgb)) for symbol, rgb in raw_colors.items())
        background = tuple(int(channel) for channel in value.get("background", (16, 29, 61)))
        return cls(
            colors=colors,  # type: ignore[arg-type]
            background=background,  # type: ignore[arg-type]
            empty_distance_factor=float(value.get("empty_distance_factor", 0.58)),
            unknown_distance=float(value.get("unknown_distance", 115.0)),
        )

    def classify(self, rgb: RGB) -> tuple[str | None, float]:
        """Return ``(symbol, confidence)``; ``None`` denotes empty/background."""

        color_items = self.colors
        nearest_color, nearest_centroid = min(
            color_items,
            key=lambda item: _distance(rgb, item[1]),
        )
        color_distance = _distance(rgb, nearest_centroid)
        background_distance = _distance(rgb, self.background)
        # A dark sample close to the background is empty even if it is somewhat
        # closer to a blue centroid due to browser shading.
        if (
            background_distance <= color_distance * self.empty_distance_factor
            or (likely_background(rgb) and color_distance > 80.0)
        ):
            confidence = max(0.0, min(1.0, 1.0 - background_distance / 180.0))
            return None, confidence
        confidence = max(0.0, min(1.0, 1.0 - color_distance / self.unknown_distance))
        if color_distance > self.unknown_distance:
            return "?", confidence
        # If a sample lies between two calibrated colors, lower confidence.
        distances = sorted(_distance(rgb, centroid) for _, centroid in color_items)
        if len(distances) > 1 and distances[1] - distances[0] < 18:
            confidence *= 0.72
        return nearest_color, confidence


def calibrate(
    samples: Iterable[CellSample],
    *,
    num_colors: int = 4,
    symbols: Sequence[str] = DEFAULT_SYMBOLS,
) -> CalibrationProfile:
    """Derive color centroids from sampled cells with deterministic clustering."""

    if num_colors < 1:
        raise ValueError("num_colors must be positive")
    sample_values = [sample.rgb for sample in samples]
    if not sample_values:
        raise ValueError("cannot calibrate from no samples")
    backgrounds = [rgb for rgb in sample_values if likely_background(rgb)]
    playable = [rgb for rgb in sample_values if not likely_background(rgb)]
    if not playable:
        raise ValueError("calibration found only dark/background samples")
    centroids = _cluster(playable, num_colors)
    if not centroids:
        raise ValueError("calibration did not produce color centroids")
    background = _mean(backgrounds) if backgrounds else (16, 29, 61)
    return CalibrationProfile(
        colors=_match_symbols(centroids, tuple(symbols)),
        background=background,
    )
