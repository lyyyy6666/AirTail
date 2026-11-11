from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from airtail.data.schema import RawPanel, SplitBounds


SPLIT_NAMES = ("train", "validation", "test")


def chronological_split(total_steps, ratios):
    ratio_values = [float(ratios[name]) for name in SPLIT_NAMES]
    if min(ratio_values) <= 0 or not np.isclose(sum(ratio_values), 1.0):
        raise ValueError("train, validation, and test ratios must be positive and sum to one")
    train_end = int(total_steps * ratio_values[0])
    validation_end = train_end + int(total_steps * ratio_values[1])
    splits = {
        "train": SplitBounds("train", 0, train_end),
        "validation": SplitBounds("validation", train_end, validation_end),
        "test": SplitBounds("test", validation_end, total_steps),
    }
    if any(split.length <= 0 for split in splits.values()):
        raise ValueError("chronological split produced an empty interval")
    return splits


def window_starts(split, history_steps, prediction_steps):
    return window_target_anchors(split, history_steps, prediction_steps) - history_steps


def window_target_anchors(split, history_steps, prediction_steps):
    if history_steps <= 0 or prediction_steps <= 0:
        raise ValueError("history_steps and prediction_steps must be positive")
    first_target = max(split.start, history_steps)
    last_target = split.end - prediction_steps
    if last_target < first_target:
        return np.empty(0, dtype=np.int64)
    return np.arange(first_target, last_target + 1, dtype=np.int64)


class FiniteStandardScaler:
    def __init__(self, mean, std):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)

    @classmethod
    def fit(cls, values, observed_mask, train_split):
        train_values = values[train_split.start : train_split.end]
        train_mask = observed_mask[train_split.start : train_split.end]
        means = np.empty(values.shape[-1], dtype=np.float32)
        stds = np.empty(values.shape[-1], dtype=np.float32)
        for feature in range(values.shape[-1]):
            finite_values = train_values[..., feature][train_mask[..., feature]]
            if finite_values.size == 0:
                raise ValueError(f"feature {feature} has no finite training observations")
            means[feature] = finite_values.mean(dtype=np.float64)
            feature_std = finite_values.std(dtype=np.float64)
            stds[feature] = feature_std if feature_std > 0 else 1.0
        return cls(means, stds)

    def transform(self, values, observed_mask):
        transformed = (np.asarray(values, dtype=np.float32) - self.mean) / self.std
        transformed[~observed_mask] = 0.0
        return transformed.astype(np.float32)

    def inverse_numpy(self, values):
        return np.asarray(values) * self.std + self.mean

    def inverse_torch(self, values):
        mean = torch.as_tensor(self.mean, dtype=values.dtype, device=values.device)
        std = torch.as_tensor(self.std, dtype=values.dtype, device=values.device)
        return values * std + mean


def calendar_time_features(timestamps, num_stations):
    features = np.empty((len(timestamps), 2), dtype=np.float32)
    for index, timestamp in enumerate(timestamps.astype("datetime64[s]").astype(np.int64)):
        value = datetime.fromtimestamp(int(timestamp), tz=timezone.utc)
        features[index, 0] = (
            value.hour + value.minute / 60.0 + value.second / 3600.0
        ) / 24.0
        features[index, 1] = value.weekday()
    return np.broadcast_to(features[:, None, :], (len(timestamps), num_stations, 2)).copy()


class UnifiedWindowDataset(Dataset):
    def __init__(
        self,
        normalized_values,
        observed_mask,
        time_features,
        timestamps,
        split,
        history_steps,
        prediction_steps,
        target_anchors=None,
    ):
        self.normalized_values = normalized_values
        self.observed_mask = observed_mask
        self.time_features = time_features
        self.timestamps = timestamps
        self.timestamp_seconds = timestamps.astype("datetime64[s]").astype(np.int64)
        self.split = split
        self.history_steps = history_steps
        self.prediction_steps = prediction_steps
        if target_anchors is None:
            target_anchors = window_target_anchors(split, history_steps, prediction_steps)
        self.target_anchors = np.asarray(target_anchors, dtype=np.int64)
        self._validate_target_anchors()
        self.starts = self.target_anchors - history_steps

    def _validate_target_anchors(self):
        anchors = self.target_anchors
        if anchors.ndim != 1:
            raise ValueError("target_anchors must be one-dimensional")
        if anchors.size > 1 and np.any(np.diff(anchors) <= 0):
            raise ValueError("target_anchors must be strictly increasing")
        if anchors.size == 0:
            return
        if anchors[0] - self.history_steps < 0:
            raise ValueError("target anchor does not have enough preceding history")
        if anchors[0] < self.split.start:
            raise ValueError("target anchor precedes its split")
        if anchors[-1] + self.prediction_steps > self.split.end:
            raise ValueError("target interval crosses its split end")

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, index):
        history_start = int(self.starts[index])
        target_start = history_start + self.history_steps
        history_end = target_start
        target_end = target_start + self.prediction_steps
        return {
            "history": torch.from_numpy(self.normalized_values[history_start:history_end]),
            "target": torch.from_numpy(self.normalized_values[target_start:target_end]),
            "history_mask": torch.from_numpy(self.observed_mask[history_start:history_end]),
            "target_mask": torch.from_numpy(self.observed_mask[target_start:target_end]),
            "history_time_features": torch.from_numpy(self.time_features[history_start:history_end]),
            "history_indices": torch.arange(history_start, history_end, dtype=torch.int64),
            "target_indices": torch.arange(target_start, target_end, dtype=torch.int64),
            "history_timestamps": torch.from_numpy(
                self.timestamp_seconds[history_start:history_end].copy()
            ),
            "target_timestamps": torch.from_numpy(
                self.timestamp_seconds[target_start:target_end].copy()
            ),
            "target_anchor": torch.tensor(target_start, dtype=torch.int64),
        }


@dataclass
class DataBundle:
    panel: RawPanel
    splits: dict[str, SplitBounds]
    scaler: FiniteStandardScaler
    datasets: dict[str, UnifiedWindowDataset]
    loaders: dict[str, DataLoader]


def build_loaders(datasets, loader_config):
    unknown = set(datasets) - set(SPLIT_NAMES)
    if unknown or not datasets:
        raise ValueError(f"invalid dataset splits for loaders: {sorted(unknown)}")
    loaders = {}
    for name, dataset in datasets.items():
        is_train = name == "train"
        generator = None
        if is_train:
            generator = torch.Generator()
            generator.manual_seed(int(loader_config["seed"]))
        loaders[name] = DataLoader(
            dataset,
            batch_size=loader_config["batch_size"],
            shuffle=is_train,
            drop_last=loader_config["drop_last_train"] if is_train else False,
            num_workers=loader_config["num_workers"],
            generator=generator,
        )
    return loaders


def build_protocol(
    panel, protocol, loader_overrides=None, included_splits=SPLIT_NAMES
):
    panel.validate()
    splits = chronological_split(panel.values.shape[0], protocol["split_ratios"])
    included_splits = tuple(included_splits)
    if not included_splits or len(set(included_splits)) != len(included_splits):
        raise ValueError("included_splits must contain unique split names")
    unknown_splits = set(included_splits) - set(SPLIT_NAMES)
    if unknown_splits:
        raise ValueError(f"unknown included split(s): {sorted(unknown_splits)}")
    history_steps = int(protocol["window"]["history_steps"])
    prediction_steps = int(protocol["window"]["prediction_steps"])
    anchor_prediction_steps = int(
        protocol["window"].get("anchor_prediction_steps", prediction_steps)
    )
    if anchor_prediction_steps < prediction_steps:
        raise ValueError("anchor_prediction_steps cannot be shorter than prediction_steps")
    interval_hours = panel.sampling_interval_hours
    if protocol["window"]["history_hours"] != history_steps * interval_hours:
        raise ValueError("history_hours does not match history_steps * sampling interval")
    if protocol["window"]["prediction_hours"] != prediction_steps * interval_hours:
        raise ValueError("prediction_hours does not match prediction_steps * sampling interval")
    scaler = FiniteStandardScaler.fit(panel.values, panel.observed_mask, splits["train"])
    selected_end = max(splits[name].end for name in included_splits)
    selected_mask = panel.observed_mask[:selected_end]
    selected_timestamps = panel.timestamps[:selected_end]
    normalized = scaler.transform(panel.values[:selected_end], selected_mask)
    time_features = calendar_time_features(
        selected_timestamps, panel.values.shape[1]
    )
    datasets = {}
    for name, split in splits.items():
        if name not in included_splits:
            continue
        anchors = window_target_anchors(
            split, history_steps, anchor_prediction_steps
        )
        datasets[name] = UnifiedWindowDataset(
            normalized,
            selected_mask,
            time_features,
            selected_timestamps,
            split,
            history_steps,
            prediction_steps,
            target_anchors=anchors,
        )
    loader_config = copy.deepcopy(protocol["loader"])
    if loader_overrides is not None:
        unknown = set(loader_overrides) - set(loader_config)
        if unknown:
            raise ValueError(f"unknown loader override(s): {sorted(unknown)}")
        loader_config.update(loader_overrides)
    loaders = build_loaders(datasets, loader_config)
    return DataBundle(panel, splits, scaler, datasets, loaders)


def build_paired_history_datasets(bundle, history_steps):
    history_steps = tuple(sorted({int(value) for value in history_steps}))
    if not history_steps or history_steps[0] <= 0:
        raise ValueError("history_steps must contain positive lengths")
    longest_history = history_steps[-1]
    paired = {length: {} for length in history_steps}
    for split_name, source in bundle.datasets.items():
        split = bundle.splits[split_name]
        anchors = window_target_anchors(
            split, longest_history, source.prediction_steps
        )
        for length in history_steps:
            paired[length][split_name] = UnifiedWindowDataset(
                source.normalized_values,
                source.observed_mask,
                source.time_features,
                source.timestamps,
                split,
                length,
                source.prediction_steps,
                target_anchors=anchors,
            )
    return paired


def masked_regression_metrics(prediction, target, target_mask, scaler):
    prediction_raw = scaler.inverse_numpy(prediction)
    target_raw = scaler.inverse_numpy(target)
    valid = np.asarray(target_mask)
    if valid.dtype != np.bool_ or valid.shape != prediction_raw.shape:
        raise ValueError("target_mask must be boolean and match prediction shape")
    if target_raw.shape != prediction_raw.shape:
        raise ValueError("prediction and target shapes must match")
    if not valid.any():
        raise ValueError("no valid target observations are available for evaluation")
    if not np.isfinite(prediction_raw[valid]).all():
        raise ValueError("prediction contains non-finite values at valid targets")
    if not np.isfinite(target_raw[valid]).all():
        raise ValueError("target contains non-finite values at valid targets")
    error = prediction_raw[valid] - target_raw[valid]
    absolute_error = np.abs(error)
    denominator = np.abs(target_raw[valid])
    nonzero = denominator > 0
    mape = float(np.mean(absolute_error[nonzero] / denominator[nonzero])) if nonzero.any() else float("nan")
    smape_denominator = np.abs(prediction_raw[valid]) + denominator
    smape_terms = np.zeros_like(absolute_error, dtype=np.float64)
    smape_nonzero = smape_denominator > 0
    smape_terms[smape_nonzero] = absolute_error[smape_nonzero] / smape_denominator[smape_nonzero]
    return {
        "count": int(valid.sum()),
        "mae": float(absolute_error.mean()),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mape": mape,
        "mape_count": int(nonzero.sum()),
        "smape": float(smape_terms.mean()),
    }
