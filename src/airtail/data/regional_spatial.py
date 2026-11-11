"""Historical meteorology and spatial inputs for regional hourly datasets."""

import copy
import csv
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch

from airtail.data.adapters import _parse_netcdf_hour_units
from airtail.data.bthsa_spatial import (
    _optional_text_attribute,
    BTHSASpatialBundle,
    BTHSASpatialWindowDataset,
    project_lonlat_to_local_km,
    projected_knn,
    stack_wind_uv,
)
from airtail.data.factory import PROJECT_ROOT, load_config, load_manifest, check_data_files
from airtail.data.netcdf import NetCDFReader
from airtail.data.protocol import (
    FiniteStandardScaler,
    build_loaders,
    calendar_time_features,
    chronological_split,
)


def _path(value):
    value = Path(value)
    return value if value.is_absolute() else PROJECT_ROOT / value


def _load_panel(config, selected_end, history_names):
    source = config["source"]
    target_name = config["selection"]["target_variable"]
    if config["dataset"] == "YRD":
        data_path = _path(source["data_path"])
        with NetCDFReader(
            data_path,
        ) as reader:
            target = reader.read_float(target_name, selected_end)[..., None]
            met = np.stack(
                [reader.read_float(name, selected_end) for name in history_names], axis=-1
            )
            station_ids = reader.read_strings("station")
            time_offsets = reader.read_double("time")[:selected_end]
            time_base = _parse_netcdf_hour_units(reader.text_attribute("time", "units"))
            units = tuple(_optional_text_attribute(reader, name, "units") for name in history_names)
        timestamps = np.asarray(
            [np.datetime64(time_base + timedelta(hours=float(offset)), "s") for offset in time_offsets]
        )
        with _path(source["stations_path"]).open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if tuple(str(row["station_id"]) for row in rows) != station_ids:
            raise ValueError("YRD station metadata order does not match NetCDF station axis")
        longitude = np.asarray([float(row["lon"]) for row in rows])
        latitude = np.asarray([float(row["lat"]) for row in rows])
        files = (data_path, _path(source["stations_path"]))
    elif config["dataset"] == "GansuAir":
        data_path = _path(source["raw_data_path"])
        records = {}
        with data_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or not {"stationID", target_name, "O3"}.issubset(reader.fieldnames):
                raise ValueError("Gansu raw CSV must contain stationID, PM2.5, and O3")
            for row in reader:
                station = row["stationID"].strip()
                if not station.isdigit():
                    raise ValueError(f"invalid raw Gansu stationID: {station!r}")
                records.setdefault(station, []).append(row)
        station_ids = tuple(sorted(records, key=int))
        if any(len(records[name]) != selected_end for name in station_ids):
            raise ValueError("raw Gansu station series do not match the timestamp length")
        feature_columns = {target_name: [], "O3": []}
        for station in station_ids:
            for name in feature_columns:
                feature_columns[name].append(
                    np.asarray([float(row[name]) for row in records[station][:selected_end]], dtype=np.float32)
                )
        target = np.stack(feature_columns.pop(target_name), axis=1)[..., None]
        observed_met = {name: np.stack(values, axis=1) for name, values in feature_columns.items() if name == "O3"}
        met_path = _path(source["meteorology_path"])
        met_columns = json.loads(_path(source["meteorology_columns_path"]).read_text(encoding="utf-8"))
        met_station_ids = tuple(json.loads(_path(source["meteorology_station_order_path"]).read_text(encoding="utf-8")))
        if met_station_ids != tuple(source["meteorology_station_ids"]):
            raise ValueError("Gansu meteorology station order does not match its recorded contract")
        if station_ids != tuple(str(i) for i in range(len(station_ids))):
            raise ValueError("Gansu air-quality stationID order must be numeric 0..N-1")
        coordinates_path = _path(source["coordinates_path"])
        coords_station_ids = tuple(json.loads(_path(source["coordinates_station_order_path"]).read_text(encoding="utf-8")))
        if coords_station_ids != met_station_ids or len(coords_station_ids) != len(station_ids):
            raise ValueError("Gansu coordinate station order does not match air-quality panel")
        coordinates = np.load(coordinates_path).astype(np.float64)
        if coordinates.shape != (len(station_ids), 2):
            raise ValueError("Gansu coordinates must have shape [stations,2]")
        longitude, latitude = coordinates[:, 0], coordinates[:, 1]
        met_tensor = np.load(met_path, mmap_mode="r")
        if met_tensor.shape[:2] != (selected_end, len(station_ids)):
            raise ValueError("Gansu meteorology tensor axes do not match raw panel")
        if len(set(met_columns)) != len(met_columns):
            raise ValueError("Gansu meteorology column names must be unique")
        met_parts = []
        for name in history_names:
            if name == "O3":
                met_parts.append(observed_met[name][..., None])
            else:
                if name not in met_columns:
                    raise ValueError(f"Gansu meteorology tensor has no {name!r} field")
                met_parts.append(np.asarray(met_tensor[:selected_end, :, met_columns.index(name)])[..., None])
        met = np.concatenate(met_parts, axis=-1).astype(np.float32)
        timestamps_path = _path(source["timestamps_path"])
        with timestamps_path.open("r", encoding="utf-8-sig", newline="") as handle:
            timestamps = np.asarray(
                [np.datetime64(datetime.fromisoformat(row["datetime"].strip()), "s")
                 for row in csv.DictReader(handle)]
            )[:selected_end]
        if len(timestamps) != selected_end:
            raise ValueError("Gansu timestamps do not match the selected raw panel")
        units = tuple("unknown" for _ in history_names)
        files = (data_path, met_path, coordinates_path, timestamps_path)
    else:
        raise ValueError(f"unsupported regional spatial dataset: {config['dataset']}")
    return target.astype(np.float32), met.astype(np.float32), timestamps, station_ids, longitude, latitude, units, files


def build_regional_spatial_bundle(
    data_config_path,
    *,
    check_files=True,
    loader_overrides=None,
    included_splits=("train", "validation"),
    future_met_variables=(),
):
    config = load_config(data_config_path)
    manifest = load_manifest(config)
    if check_files:
        check_data_files(manifest)
    spatial = config["qc_mssm_spatial"]
    history_names = tuple(spatial["history_met_variables"])
    wind_names = tuple(spatial["wind_uv_variables"])
    future_names = tuple(future_met_variables)
    if any(name not in history_names for name in future_names):
        raise ValueError("teacher future meteorology must be selected from historical meteorology")
    total_steps = manifest["data"].get("time_steps", manifest["data"]["raw_variable_shape"][0])
    splits = chronological_split(total_steps, config["protocol"]["split_ratios"])
    selected_end = max(splits[name].end for name in included_splits)
    target, met, timestamps, station_ids, longitude, latitude, units, files = _load_panel(
        config, selected_end, history_names
    )
    target_mask, met_mask = np.isfinite(target), np.isfinite(met)
    target_scaler = FiniteStandardScaler.fit(target, target_mask, splits["train"])
    met_scaler = FiniteStandardScaler.fit(met, met_mask, splits["train"])
    normalized_target = target_scaler.transform(target, target_mask)
    normalized_met = met_scaler.transform(met, met_mask)
    wind_indices = [history_names.index(name) for name in wind_names]
    raw_wind = stack_wind_uv(met[..., wind_indices[0]], met[..., wind_indices[1]])
    future_indices = [history_names.index(name) for name in future_names]
    future_met = normalized_met[..., future_indices] if future_indices else None
    future_mask = met_mask[..., future_indices] if future_indices else None
    projection = copy.deepcopy(spatial["projection"])
    coords = project_lonlat_to_local_km(
        longitude=longitude, latitude=latitude,
        center_longitude=projection["center_longitude"],
        center_latitude=projection["center_latitude"],
        earth_radius_km=projection["earth_radius_km"],
    )
    neighbor_index, edge_distances = projected_knn(coords, spatial["neighbor_count"])
    time_features = calendar_time_features(timestamps, len(station_ids))
    window = config["protocol"]["window"]
    datasets = {
        name: BTHSASpatialWindowDataset(
            normalized_met, raw_wind, normalized_target, target_mask,
            time_features, timestamps, split, int(window["history_steps"]),
            int(window["prediction_steps"]), future_met=future_met,
            future_met_observed_mask=future_mask,
        )
        for name, split in splits.items() if name in included_splits
    }
    loader_config = copy.deepcopy(config["protocol"]["loader"])
    if loader_overrides:
        loader_config.update(loader_overrides)
    loaders = build_loaders(datasets, loader_config)
    percentiles = np.percentile(edge_distances, [0, 25, 50, 75, 100])
    return BTHSASpatialBundle(
        station_ids=station_ids, target_feature_name=config["selection"]["target_variable"],
        target_feature_unit="ug/m3", met_feature_names=history_names,
        met_feature_units=units, future_met_feature_names=future_names,
        future_met_feature_units=tuple(units[history_names.index(name)] for name in future_names),
        wind_feature_names=wind_names, splits=splits, target_scaler=target_scaler,
        met_scaler=met_scaler, datasets=datasets, loaders=loaders,
        lonlat_deg=torch.from_numpy(np.stack((longitude, latitude), axis=-1).astype(np.float32)),
        coords_km=torch.from_numpy(coords),
        neighbor_index=torch.from_numpy(neighbor_index),
        projection=projection,
        neighbor_distance_km={"min": float(percentiles[0]), "p25": float(percentiles[1]),
                              "median": float(percentiles[2]), "p75": float(percentiles[3]),
                              "max": float(percentiles[4])},
    )
