import copy
import csv
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch

from airtail.data.adapters import _parse_netcdf_hour_units
from airtail.data.factory import (
    PROJECT_ROOT,
    load_config,
    load_manifest,
    check_data_files,
)
from airtail.data.netcdf import NetCDFReader
from airtail.data.protocol import (
    SPLIT_NAMES,
    FiniteStandardScaler,
    UnifiedWindowDataset,
    build_loaders,
    calendar_time_features,
    chronological_split,
)


def project_lonlat_to_local_km(
    *,
    longitude,
    latitude,
    center_longitude,
    center_latitude,
    earth_radius_km,
    longitude_range=None,
    latitude_range=None,
):
    longitude = np.asarray(longitude, dtype=np.float64)
    latitude = np.asarray(latitude, dtype=np.float64)
    if longitude.ndim != 1 or latitude.shape != longitude.shape:
        raise ValueError("longitude and latitude must be matching one-dimensional arrays")
    if not np.isfinite(longitude).all() or not np.isfinite(latitude).all():
        raise ValueError("longitude and latitude must be finite")
    if longitude_range is not None and (
        longitude.min() < longitude_range[0] or longitude.max() > longitude_range[1]
    ):
        raise ValueError("longitude values are outside the configured BTHSA range")
    if latitude_range is not None and (
        latitude.min() < latitude_range[0] or latitude.max() > latitude_range[1]
    ):
        raise ValueError("latitude values are outside the configured BTHSA range")
    if not -180 <= center_longitude <= 180 or not -90 <= center_latitude <= 90:
        raise ValueError("projection center is not a valid longitude/latitude")
    if earth_radius_km <= 0:
        raise ValueError("earth_radius_km must be positive")

    lon_offset = np.deg2rad(longitude - center_longitude)
    lat_offset = np.deg2rad(latitude - center_latitude)
    east_km = earth_radius_km * np.cos(np.deg2rad(center_latitude)) * lon_offset
    north_km = earth_radius_km * lat_offset
    return np.stack((east_km, north_km), axis=-1).astype(np.float32)


def projected_knn(coords_km, neighbor_count):
    coords_km = np.asarray(coords_km, dtype=np.float64)
    if coords_km.ndim != 2 or coords_km.shape[1] != 2:
        raise ValueError("coords_km must have shape [N,2]")
    if not np.isfinite(coords_km).all():
        raise ValueError("coords_km must be finite")
    neighbor_count = int(neighbor_count)
    if not 0 < neighbor_count < coords_km.shape[0]:
        raise ValueError("neighbor_count must be in [1,N)")
    distances = np.linalg.norm(
        coords_km[:, None, :] - coords_km[None, :, :], axis=-1
    )
    np.fill_diagonal(distances, np.inf)
    neighbor_index = np.argsort(distances, axis=1, kind="stable")[:, :neighbor_count]
    edge_distances = distances[
        np.arange(coords_km.shape[0])[:, None], neighbor_index
    ]
    return neighbor_index.astype(np.int64), edge_distances.astype(np.float32)


def stack_wind_uv(u_east, v_north):
    u_east = np.asarray(u_east, dtype=np.float32)
    v_north = np.asarray(v_north, dtype=np.float32)
    if u_east.shape != v_north.shape:
        raise ValueError("u_east and v_north must have matching shapes")
    wind_uv = np.stack((u_east, v_north), axis=-1)
    if not np.isfinite(wind_uv).all():
        raise ValueError("physical wind components must be finite")
    return wind_uv


class BTHSASpatialWindowDataset(UnifiedWindowDataset):
    def __init__(
        self,
        normalized_met,
        raw_wind_uv,
        *args,
        future_met=None,
        future_met_observed_mask=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if normalized_met.shape[:2] != self.normalized_values.shape[:2]:
            raise ValueError("normalized meteorology must align with target time/station axes")
        if raw_wind_uv.shape != normalized_met.shape[:2] + (2,):
            raise ValueError("raw_wind_uv must have shape [T,N,2]")
        if (future_met is None) != (future_met_observed_mask is None):
            raise ValueError(
                "future_met and future_met_observed_mask must be provided together"
            )
        if future_met is not None:
            if future_met.shape[:2] != normalized_met.shape[:2]:
                raise ValueError("future meteorology must align with target time/station axes")
            if future_met_observed_mask.shape != future_met.shape:
                raise ValueError("future_met_observed_mask must match future meteorology")
            if future_met_observed_mask.dtype != np.bool_:
                raise ValueError("future_met_observed_mask must be boolean")
        self.normalized_met = normalized_met
        self.raw_wind_uv = raw_wind_uv
        self.future_met = future_met
        self.future_met_observed_mask = future_met_observed_mask

    def __getitem__(self, index):
        item = super().__getitem__(index)
        history_start = int(self.starts[index])
        history_end = history_start + self.history_steps
        item["history_met"] = torch.from_numpy(
            self.normalized_met[history_start:history_end]
        )
        item["history_wind_uv"] = torch.from_numpy(
            self.raw_wind_uv[history_start:history_end]
        )
        if self.future_met is not None:
            target_start = history_end
            target_end = target_start + self.prediction_steps
            item["future_met"] = torch.from_numpy(
                self.future_met[target_start:target_end]
            )
            item["future_met_mask"] = torch.from_numpy(
                self.future_met_observed_mask[target_start:target_end].all(
                    axis=-1, keepdims=True
                )
            )
        return item


@dataclass
class BTHSASpatialBundle:
    station_ids: tuple
    target_feature_name: str
    target_feature_unit: str
    met_feature_names: tuple
    met_feature_units: tuple
    future_met_feature_names: tuple
    future_met_feature_units: tuple
    wind_feature_names: tuple
    splits: dict
    target_scaler: FiniteStandardScaler
    met_scaler: FiniteStandardScaler
    datasets: dict
    loaders: dict
    lonlat_deg: torch.Tensor
    coords_km: torch.Tensor
    neighbor_index: torch.Tensor
    projection: dict
    neighbor_distance_km: dict


def _read_station_metadata(path, expected_station_ids, spatial_config):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    longitude_column = spatial_config["coordinate_columns"]["longitude"]
    latitude_column = spatial_config["coordinate_columns"]["latitude"]
    required = {"station_id", longitude_column, latitude_column}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("BTHSA station CSV is missing named coordinate columns")
    station_ids = tuple(str(row["station_id"]) for row in rows)
    if station_ids != expected_station_ids:
        raise ValueError("station CSV order does not match the NetCDF station axis")
    longitude = np.asarray([float(row[longitude_column]) for row in rows])
    latitude = np.asarray([float(row[latitude_column]) for row in rows])
    return longitude, latitude


def _distance_summary(edge_distances):
    values = np.percentile(edge_distances, [0, 25, 50, 75, 100])
    return {
        "min": float(values[0]),
        "p25": float(values[1]),
        "median": float(values[2]),
        "p75": float(values[3]),
        "max": float(values[4]),
    }


def _optional_text_attribute(reader, variable, attribute):
    try:
        return reader.text_attribute(variable, attribute)
    except RuntimeError:
        return None


def build_bthsa_spatial_bundle(
    data_config_path="configs/unified_data/bthsa.json",
    check_files=True,
    loader_overrides=None,
    included_splits=SPLIT_NAMES,
    future_met_variables=None,
):
    config = load_config(data_config_path)
    if config["dataset"] != "BTHSA":
        raise ValueError("BTHSA spatial loading requires the BTHSA data config")
    manifest = load_manifest(config)
    if check_files:
        check_data_files(manifest)
    spatial = config["qc_mssm_spatial"]
    met_names = tuple(spatial["history_met_variables"])
    future_met_names = tuple(future_met_variables or ())
    if len(set(future_met_names)) != len(future_met_names):
        raise ValueError("future meteorology variables must be unique")
    unknown_future = set(future_met_names) - set(met_names)
    if unknown_future:
        raise ValueError(
            f"future meteorology variables are not in the BTHSA panel: {sorted(unknown_future)}"
        )
    forbidden_future = {config["selection"]["target_variable"], "O3"}
    leaked_future = set(future_met_names) & forbidden_future
    if leaked_future:
        raise ValueError(
            f"future meteorology cannot contain pollutant variables: {sorted(leaked_future)}"
        )
    wind_names = tuple(spatial["wind_uv_variables"])
    if wind_names != ("u100", "v100"):
        raise ValueError("BTHSA physical wind order must be [u100,v100]")
    if any(name not in met_names for name in wind_names):
        raise ValueError("physical wind variables must also be present in history_met")

    included_splits = tuple(included_splits)
    if not included_splits or set(included_splits) - set(SPLIT_NAMES):
        raise ValueError("included_splits contains an unknown split")
    source = config["source"]
    data_path = Path(source["data_path"])
    stations_path = Path(source["stations_path"])
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path
    if not stations_path.is_absolute():
        stations_path = PROJECT_ROOT / stations_path

    with NetCDFReader(
        data_path,
    ) as reader:
        total_steps, num_stations = reader.variable_shape(
            config["selection"]["target_variable"]
        )
        splits = chronological_split(total_steps, config["protocol"]["split_ratios"])
        selected_end = max(splits[name].end for name in included_splits)
        target = reader.read_float(
            config["selection"]["target_variable"], selected_end
        )[..., None]
        met = np.empty((selected_end, num_stations, len(met_names)), dtype=np.float32)
        for channel, name in enumerate(met_names):
            met[..., channel] = reader.read_float(name, selected_end)
        target_unit = _optional_text_attribute(
            reader, config["selection"]["target_variable"], "units"
        )
        met_units = tuple(
            _optional_text_attribute(reader, name, "units") for name in met_names
        )
        station_ids = reader.read_strings("station")
        time_offsets = reader.read_double("time")[:selected_end]
        time_base = _parse_netcdf_hour_units(reader.text_attribute("time", "units"))
        wind_units = tuple(reader.text_attribute(name, "units") for name in wind_names)
    if wind_units != ("m/s", "m/s"):
        raise ValueError(f"BTHSA wind components must use m/s, got {wind_units}")
    if len(station_ids) != num_stations:
        raise ValueError("NetCDF station coordinate does not match the station axis")

    timestamps = np.asarray(
        [
            np.datetime64(time_base + timedelta(hours=float(offset)), "s")
            for offset in time_offsets
        ]
    )
    target_mask = np.isfinite(target)
    met_mask = np.isfinite(met)
    target_scaler = FiniteStandardScaler.fit(target, target_mask, splits["train"])
    met_scaler = FiniteStandardScaler.fit(met, met_mask, splits["train"])
    normalized_target = target_scaler.transform(target, target_mask)
    wind_indices = [met_names.index(name) for name in wind_names]
    raw_wind_uv = stack_wind_uv(
        met[..., wind_indices[0]], met[..., wind_indices[1]]
    )
    normalized_met = met_scaler.transform(met, met_mask)
    future_indices = [met_names.index(name) for name in future_met_names]
    future_met = normalized_met[..., future_indices] if future_indices else None
    future_met_mask = met_mask[..., future_indices] if future_indices else None
    time_features = calendar_time_features(timestamps, num_stations)

    longitude, latitude = _read_station_metadata(
        stations_path, station_ids, spatial
    )
    projection = copy.deepcopy(spatial["projection"])
    if projection["method"] != "local_equirectangular":
        raise ValueError("unsupported BTHSA projection method")
    if projection["input_order"] != ["longitude", "latitude"]:
        raise ValueError("BTHSA projection input order must be [longitude,latitude]")
    if projection["output_order"] != ["east", "north"]:
        raise ValueError("BTHSA projection output order must be [east,north]")
    if projection["output_units"] != "km":
        raise ValueError("BTHSA projected coordinates must use km")
    coords_km = project_lonlat_to_local_km(
        longitude=longitude,
        latitude=latitude,
        center_longitude=projection["center_longitude"],
        center_latitude=projection["center_latitude"],
        earth_radius_km=projection["earth_radius_km"],
        longitude_range=projection["longitude_range"],
        latitude_range=projection["latitude_range"],
    )
    neighbor_index, edge_distances = projected_knn(
        coords_km, spatial["neighbor_count"]
    )

    window = config["protocol"]["window"]
    datasets = {
        name: BTHSASpatialWindowDataset(
            normalized_met,
            raw_wind_uv,
            normalized_target,
            target_mask,
            time_features,
            timestamps,
            split,
            int(window["history_steps"]),
            int(window["prediction_steps"]),
            future_met=future_met,
            future_met_observed_mask=future_met_mask,
        )
        for name, split in splits.items()
        if name in included_splits
    }
    loader_config = copy.deepcopy(config["protocol"]["loader"])
    if loader_overrides is not None:
        unknown = set(loader_overrides) - set(loader_config)
        if unknown:
            raise ValueError(f"unknown loader override(s): {sorted(unknown)}")
        loader_config.update(loader_overrides)
    loaders = build_loaders(datasets, loader_config)
    return BTHSASpatialBundle(
        station_ids=station_ids,
        target_feature_name=config["selection"]["target_variable"],
        target_feature_unit=target_unit,
        met_feature_names=met_names,
        met_feature_units=met_units,
        future_met_feature_names=future_met_names,
        future_met_feature_units=tuple(
            met_units[met_names.index(name)] for name in future_met_names
        ),
        wind_feature_names=wind_names,
        splits=splits,
        target_scaler=target_scaler,
        met_scaler=met_scaler,
        datasets=datasets,
        loaders=loaders,
        lonlat_deg=torch.from_numpy(
            np.stack((longitude, latitude), axis=-1).astype(np.float32)
        ),
        coords_km=torch.from_numpy(coords_km),
        neighbor_index=torch.from_numpy(neighbor_index),
        projection=projection,
        neighbor_distance_km=_distance_summary(edge_distances),
    )


def prepare_bthsa_qc_batch(batch, bundle, device, taus):
    required = {
        "history",
        "history_mask",
        "history_met",
        "history_wind_uv",
        "history_time_features",
        "target",
        "target_mask",
        "history_indices",
        "target_anchor",
    }
    missing = sorted(required - set(batch))
    if missing:
        raise ValueError(f"BTHSA QC batch is missing keys: {missing}")
    history = batch["history"]
    history_met = batch["history_met"]
    history_wind_uv = batch["history_wind_uv"]
    future_met = batch.get("future_met")
    future_met_mask = batch.get("future_met_mask")
    batch_size, length, stations, _ = history.shape
    if history_met.shape != (
        batch_size,
        length,
        stations,
        len(bundle.met_feature_names),
    ):
        raise ValueError("history_met has an invalid BTHSA QC shape")
    if history_wind_uv.shape != (batch_size, length, stations, 2):
        raise ValueError("history_wind_uv must have shape [B,L,N,2]")
    if not torch.isfinite(history_met).all() or not torch.isfinite(history_wind_uv).all():
        raise ValueError("BTHSA meteorology and physical wind must be finite")
    if (future_met is None) != (future_met_mask is None):
        raise ValueError("future_met and future_met_mask must be provided together")
    if future_met is not None:
        if future_met.shape != (
            batch_size,
            batch["target"].shape[1],
            stations,
            len(bundle.future_met_feature_names),
        ):
            raise ValueError("future_met has an invalid BTHSA QC shape")
        if future_met_mask.shape != future_met.shape[:3] + (1,):
            raise ValueError("future_met_mask must have shape [B,H,N,1]")
        if future_met_mask.dtype != torch.bool:
            raise ValueError("future_met_mask must be boolean")
        if not torch.isfinite(future_met).all():
            raise ValueError("future_met must be finite after train-only scaling")
    time_features = batch["history_time_features"]
    if time_features.shape != (batch_size, length, stations, 2):
        raise ValueError("history_time_features has an invalid shape")
    shared_time = time_features[:, :, :1].expand_as(time_features)
    if not torch.equal(time_features, shared_time):
        raise ValueError("calendar features must be identical across stations")
    if not torch.equal(batch["history_indices"][:, -1] + 1, batch["target_anchor"]):
        raise ValueError("history must end immediately before the target anchor")
    if bundle.coords_km.shape != (stations, 2):
        raise ValueError("projected coordinates do not match the batch station axis")
    if bundle.neighbor_index.shape[0] != stations:
        raise ValueError("neighbor indices do not match the batch station axis")
    return {
        "model_inputs": {
            "history_values": history.to(device),
            "history_mask": batch["history_mask"].to(device),
            "history_met": history_met.to(device),
            "history_wind_uv": history_wind_uv.to(device),
            "history_time_features": time_features[:, :, 0].to(device),
            "coords": bundle.coords_km.to(device),
            "neighbor_index": bundle.neighbor_index.to(device),
            "taus": taus.to(device),
        },
        "target": batch["target"].to(device),
        "target_mask": batch["target_mask"].to(device),
        **(
            {
                "future_met": future_met.to(device),
                "future_met_mask": future_met_mask.to(device),
            }
            if future_met is not None
            else {}
        ),
    }
