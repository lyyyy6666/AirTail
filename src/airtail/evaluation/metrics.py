import numpy as np
import torch

from airtail.evaluation.paper_metrics import PaperMetrics

from airtail.data.bthsa_spatial import prepare_bthsa_qc_batch


class QuantileMetricAccumulator:
    def __init__(self, taus, target_scaler, tau_u=None):
        self.taus = torch.as_tensor(taus, dtype=torch.float64).cpu()
        if self.taus.ndim != 1 or self.taus.numel() == 0:
            raise ValueError("evaluation taus must be a non-empty vector")
        median = torch.nonzero(self.taus == 0.5, as_tuple=False).flatten()
        if median.numel() != 1:
            raise ValueError("evaluation taus must contain exactly one 0.5 quantile")
        self.median_index = int(median.item())
        self.tau_u = None if tau_u is None else float(tau_u)
        if self.tau_u is not None and not 0.5 < self.tau_u < 1.0:
            raise ValueError("tau_u must lie strictly between 0.5 and 1")
        self.target_std = float(np.asarray(target_scaler.std).reshape(-1)[0])
        count = self.taus.numel()
        self.pinball_sum = torch.zeros(count, dtype=torch.float64)
        self.coverage_sum = torch.zeros(count, dtype=torch.float64)
        self.absolute_error_sum = 0.0
        self.squared_error_sum = 0.0
        self.crossing_count = 0
        self.valid_count = 0

    def update(self, prediction, target, target_mask):
        if prediction.ndim != 4 or prediction.shape[-1] != self.taus.numel():
            raise ValueError("quantile prediction must have shape [B,H,N,Q]")
        if target.shape != prediction.shape[:3] + (1,):
            raise ValueError("quantile target must have shape [B,H,N,1]")
        if target_mask.shape != target.shape or target_mask.dtype != torch.bool:
            raise ValueError("target_mask must be boolean and match target")
        valid = target_mask[..., 0]
        batch_valid = int(valid.sum().item())
        if batch_valid == 0:
            return
        prediction = prediction.detach().double().cpu()
        target = target.detach().double().cpu()[..., 0]
        valid = valid.detach().cpu()
        taus = self.taus.view(1, 1, 1, -1)
        error = target.unsqueeze(-1) - prediction
        pinball = torch.maximum(taus * error, (taus - 1.0) * error)
        self.pinball_sum += (pinball * valid.unsqueeze(-1)).sum(dim=(0, 1, 2))
        self.coverage_sum += (
            (target.unsqueeze(-1) <= prediction) * valid.unsqueeze(-1)
        ).sum(dim=(0, 1, 2))
        median_error = (
            prediction[..., self.median_index] - target
        ) * self.target_std
        valid_error = median_error[valid]
        self.absolute_error_sum += valid_error.abs().sum().item()
        self.squared_error_sum += valid_error.square().sum().item()
        if prediction.shape[-1] > 1:
            crossings = prediction[..., 1:] < prediction[..., :-1]
            self.crossing_count += int(
                (crossings * valid.unsqueeze(-1)).sum().item()
            )
        self.valid_count += batch_valid

    def compute(self):
        if self.valid_count == 0:
            raise ValueError("no valid targets were accumulated")
        per_tau = self.pinball_sum / self.valid_count
        coverage = self.coverage_sum / self.valid_count
        crossing_denominator = self.valid_count * max(self.taus.numel() - 1, 1)
        result = {
            "standardized_pinball": float(per_tau.mean().item()),
            "standardized_pinball_by_tau": {
                f"{tau:.6g}": float(value)
                for tau, value in zip(self.taus.tolist(), per_tau.tolist())
            },
            "original_unit_median_mae": self.absolute_error_sum / self.valid_count,
            "original_unit_median_rmse": (
                self.squared_error_sum / self.valid_count
            ) ** 0.5,
            "empirical_coverage_by_tau": {
                f"{tau:.6g}": float(value)
                for tau, value in zip(self.taus.tolist(), coverage.tolist())
            },
            "quantile_crossing_count": self.crossing_count,
            "quantile_crossing_rate": self.crossing_count / crossing_denominator,
            "valid_count": self.valid_count,
        }
        if self.tau_u is not None:
            tail = self.taus > self.tau_u
            if torch.any(tail):
                result["tail_standardized_pinball"] = float(
                    per_tau[tail].mean().item()
                )
                result["tail_mean_absolute_coverage_error"] = float(
                    (coverage[tail] - self.taus[tail]).abs().mean().item()
                )
        return result


def evaluation_taus(config, device):
    taus = torch.tensor(config["taus"]["fixed"], dtype=torch.float32, device=device)
    if taus.ndim != 1 or not torch.all(taus[1:] > taus[:-1]):
        raise ValueError("fixed evaluation taus must be strictly increasing")
    return taus




def _student_forward(student, prepared, taus):
    inputs = prepared["model_inputs"]
    output = student(
        history_values=inputs["history_values"],
        history_mask=inputs["history_mask"],
        history_met=inputs["history_met"],
        history_wind_uv=inputs["history_wind_uv"],
        history_time_features=inputs["history_time_features"],
        coords=inputs["coords"],
        neighbor_index=inputs["neighbor_index"],
        taus=taus,
    )
    # Relation-KD V2 returns (quantile_prediction, relation_diagnostics),
    # while the shared evaluator consumes the quantile prediction object.
    if isinstance(output, tuple):
        if len(output) != 2:
            raise ValueError("Student forward tuple must contain prediction and relation output")
        output = output[0]
    return output


def evaluate_qc_model(
    model,
    loader,
    bundle,
    device,
    config,
    stage,
    max_batches=None,
    *,
    paper_metrics=False,
):
    taus = evaluation_taus(config, device)
    metrics = QuantileMetricAccumulator(
        taus.cpu(), bundle.target_scaler, tau_u=config["head"]["tau_u"]
    )
    manuscript = PaperMetrics(config["taus"]["fixed"], bundle.target_scaler) if paper_metrics else None
    model.eval()
    batch_count = 0
    with torch.no_grad():
        for batch in loader:
            prepared = prepare_bthsa_qc_batch(batch, bundle, device, taus)
            output = _student_forward(model, prepared, taus)
            expected = (
                prepared["target"].shape[0],
                config["horizon"],
                config["num_nodes"],
                taus.numel(),
            )
            if output.quantiles.shape != expected:
                raise ValueError(
                    f"QC output shape {tuple(output.quantiles.shape)} does not match {expected}"
                )
            metrics.update(
                output.quantiles, prepared["target"], prepared["target_mask"]
            )
            if manuscript is not None:
                manuscript.update(output.quantiles, prepared["target"], prepared["target_mask"])
            batch_count += 1
            if max_batches is not None and batch_count >= max_batches:
                break
    if batch_count == 0:
        raise ValueError("evaluation loader is empty")
    result = metrics.compute()
    result["batches"] = batch_count
    result["stage"] = stage
    result["metric_spaces"] = {
        "pinball": "standardized_target_space",
        "median_mae_rmse": "original_target_unit",
    }
    if manuscript is not None:
        result["manuscript_metrics"] = manuscript.compute()
    return result
