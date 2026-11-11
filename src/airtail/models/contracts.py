from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class QuantileModelOutput:
    quantiles: torch.Tensor
    median: torch.Tensor
    bulk_coefficients: torch.Tensor
    threshold_u: torch.Tensor
    threshold_slope: torch.Tensor
    tail_sigma: torch.Tensor
    tail_xi: torch.Tensor
    scale_weights: torch.Tensor
    delta_summary: dict
    half_life_summary: dict
    spatial_weight_summary: dict
    diagnostics: dict | None = None
