from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SplitBounds:
    name: str
    start: int
    end: int

    @property
    def length(self):
        return self.end - self.start


@dataclass
class RawPanel:
    values: np.ndarray
    observed_mask: np.ndarray
    timestamps: np.ndarray
    station_ids: tuple[str, ...]
    feature_names: tuple[str, ...]
    target_index: int
    sampling_interval_hours: int
    source_files: tuple[Path, ...]
    adjacency: np.ndarray | None = None
    graph_station_ids: tuple[str, ...] | None = None

    def validate(self):
        if self.values.ndim != 3:
            raise ValueError(f"values must have shape [time, station, feature], got {self.values.shape}")
        if self.observed_mask.shape != self.values.shape:
            raise ValueError("observed_mask must match values shape")
        if self.observed_mask.dtype != np.bool_:
            raise ValueError("observed_mask must have boolean dtype")
        time_steps, num_stations, num_features = self.values.shape
        if self.timestamps.shape != (time_steps,):
            raise ValueError("timestamps must have shape [time]")
        if len(self.station_ids) != num_stations or len(set(self.station_ids)) != num_stations:
            raise ValueError("station_ids must be unique and match the station axis")
        if len(self.feature_names) != num_features:
            raise ValueError("feature_names must match the feature axis")
        if not 0 <= self.target_index < num_features:
            raise ValueError("target_index is outside the feature axis")
        if self.sampling_interval_hours <= 0:
            raise ValueError("sampling_interval_hours must be positive")
        expected_delta = np.timedelta64(self.sampling_interval_hours, "h")
        if time_steps > 1 and not np.all(np.diff(self.timestamps) == expected_delta):
            raise ValueError("timestamps are not regular at the configured sampling interval")
        if not np.array_equal(self.observed_mask, np.isfinite(self.values)):
            raise ValueError("observed_mask must be exactly numpy.isfinite(values)")
        if self.adjacency is not None:
            if self.adjacency.shape != (num_stations, num_stations):
                raise ValueError("adjacency shape must match the station axis")
            if self.graph_station_ids is None:
                raise ValueError("graph_station_ids are required when adjacency is provided")
            if self.graph_station_ids != self.station_ids:
                raise ValueError("graph station order does not match the panel station order")
