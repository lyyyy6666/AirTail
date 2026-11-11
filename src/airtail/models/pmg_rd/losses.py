import math
import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F


def _non_negative_finite_real(value, name):
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a finite non-negative real number")
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be a finite non-negative real number")
    return value


def _non_negative_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a non-negative integer")
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def linear_relation_distillation_weight(*, lambda_rel, global_step, warmup_steps):
    """Return the stateless linear warmup weight for relation distillation."""
    lambda_rel = _non_negative_finite_real(lambda_rel, "lambda_rel")
    global_step = _non_negative_integer(global_step, "global_step")
    warmup_steps = _non_negative_integer(warmup_steps, "warmup_steps")
    if warmup_steps == 0:
        return lambda_rel
    return lambda_rel * min(1.0, global_step / warmup_steps)


def compose_relation_kd_objective(
    pinball_loss,
    relation_loss,
    *,
    relation_weight,
):
    """Combine scalar pinball and relation losses without detaching either."""
    relation_weight = _non_negative_finite_real(relation_weight, "relation_weight")
    losses = {"pinball_loss": pinball_loss, "relation_loss": relation_loss}
    for name, loss in losses.items():
        if not isinstance(loss, torch.Tensor):
            raise TypeError(f"{name} must be a Tensor")
        if loss.ndim != 0:
            raise ValueError(f"{name} must be a scalar Tensor")
        if not loss.is_floating_point():
            raise TypeError(f"{name} must use a floating point dtype")
    if pinball_loss.dtype != relation_loss.dtype:
        raise TypeError("pinball_loss and relation_loss must share a dtype")
    if pinball_loss.device != relation_loss.device:
        raise ValueError("pinball_loss and relation_loss must share a device")
    return pinball_loss + relation_weight * relation_loss


class TailEffectiveRelationDistillationLoss(nn.Module):
    """Align tail-query relation biases after row centering."""

    def __init__(
        self,
        beta,
        query_region_indices=(1, 2),
        num_regions=3,
        num_scales=4,
    ):
        super().__init__()
        if isinstance(beta, bool) or not isinstance(beta, (int, float)):
            raise TypeError("beta must be a positive finite number")
        beta = float(beta)
        if not math.isfinite(beta) or beta <= 0.0:
            raise ValueError("beta must be a positive finite number")

        dimensions = {
            "num_regions": num_regions,
            "num_scales": num_scales,
        }
        for name, value in dimensions.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be a positive integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")

        if not isinstance(query_region_indices, (list, tuple)):
            raise TypeError("query_region_indices must be a list or tuple")
        query_region_indices = tuple(query_region_indices)
        if not query_region_indices:
            raise ValueError("query_region_indices must be non-empty")
        if any(
            isinstance(index, bool) or not isinstance(index, int)
            for index in query_region_indices
        ):
            raise TypeError("query_region_indices must contain integers")
        if len(set(query_region_indices)) != len(query_region_indices):
            raise ValueError("query_region_indices must be unique")
        if any(index < 0 or index >= num_regions for index in query_region_indices):
            raise ValueError("query_region_indices must be within the region range")

        self.beta = beta
        self.query_region_indices = query_region_indices
        self.row_centering = True
        self.num_regions = num_regions
        self.num_scales = num_scales
        self.num_tokens = num_regions * num_scales
        self.query_rows = tuple(
            region * num_scales + scale
            for region in query_region_indices
            for scale in range(num_scales)
        )

    def _validate_inputs(
        self,
        student_corr,
        teacher_corr,
        student_relation_gain,
        teacher_relation_gain,
        valid_mask,
    ):
        if not isinstance(student_corr, torch.Tensor):
            raise TypeError("student_corr must be a Tensor")
        if not isinstance(teacher_corr, torch.Tensor):
            raise TypeError("teacher_corr must be a Tensor")
        if student_corr.ndim != 6:
            raise ValueError("student_corr must be a rank-6 Tensor [B,H,N,M,K,K]")
        if teacher_corr.ndim != 6:
            raise ValueError("teacher_corr must be a rank-6 Tensor [B,H,N,M,K,K]")

        batch, horizons, stations, student_heads, rows, columns = student_corr.shape
        teacher_batch, teacher_horizons, teacher_stations, teacher_heads, teacher_rows, teacher_columns = teacher_corr.shape
        if batch <= 0 or horizons <= 0 or stations <= 0 or student_heads <= 0:
            raise ValueError("B, H, N, and M must be positive")
        if (rows, columns) != (self.num_tokens, self.num_tokens):
            raise ValueError(
                f"student_corr relation axes must have K={self.num_tokens}"
            )
        if (teacher_rows, teacher_columns) != (self.num_tokens, self.num_tokens):
            raise ValueError(
                f"teacher_corr relation axes must have K={self.num_tokens}"
            )
        if (batch, horizons, stations) != (
            teacher_batch,
            teacher_horizons,
            teacher_stations,
        ):
            raise ValueError("Student and Teacher Corr must share B, H, and N")
        if student_heads != teacher_heads:
            raise ValueError("Student and Teacher Corr head counts must match")
        if not student_corr.is_floating_point() or not teacher_corr.is_floating_point():
            raise TypeError("Student and Teacher Corr must use floating point dtypes")
        if student_corr.dtype != teacher_corr.dtype:
            raise TypeError("Student and Teacher Corr must share a dtype")
        if student_corr.device != teacher_corr.device:
            raise ValueError("Student and Teacher Corr must share a device")

        gains = {
            "student_relation_gain": (student_relation_gain, student_corr),
            "teacher_relation_gain": (teacher_relation_gain, teacher_corr),
        }
        for name, (gain, corr) in gains.items():
            if not isinstance(gain, torch.Tensor):
                raise TypeError(f"{name} must be a Tensor")
            if gain.ndim != 0:
                raise ValueError(f"{name} must be a scalar Tensor")
            if not gain.is_floating_point():
                raise TypeError(f"{name} must use a floating point dtype")
            if gain.dtype != corr.dtype:
                raise TypeError(f"{name} dtype must match its Corr")
            if gain.device != corr.device:
                raise ValueError(f"{name} device must match its Corr")

        if not isinstance(valid_mask, torch.Tensor):
            raise TypeError("valid_mask must be a Tensor")
        if valid_mask.dtype != torch.bool:
            raise TypeError("valid_mask must have bool dtype")
        if valid_mask.device != student_corr.device:
            raise ValueError("valid_mask and Corr must share a device")
        if valid_mask.shape == (batch, horizons, stations, 1):
            valid_mask = valid_mask.squeeze(-1)
        elif valid_mask.shape != (batch, horizons, stations):
            raise ValueError("valid_mask must have shape [B,H,N] or [B,H,N,1]")
        return valid_mask

    def forward_parts(
        self,
        student_corr,
        teacher_corr,
        student_relation_gain,
        teacher_relation_gain,
        valid_mask,
    ):
        valid_mask = self._validate_inputs(
            student_corr,
            teacher_corr,
            student_relation_gain,
            teacher_relation_gain,
            valid_mask,
        )

        student_bias = student_relation_gain * student_corr
        teacher_bias = (teacher_relation_gain * teacher_corr).detach()
        if self.row_centering:
            student_bias = student_bias - student_bias.mean(dim=-1, keepdim=True)
            teacher_bias = teacher_bias - teacher_bias.mean(dim=-1, keepdim=True)

        query_rows = torch.tensor(self.query_rows, device=student_corr.device)
        student_bias = student_bias.index_select(-2, query_rows)
        teacher_bias = teacher_bias.index_select(-2, query_rows)
        element_loss = F.smooth_l1_loss(
            student_bias,
            teacher_bias,
            beta=self.beta,
            reduction="none",
        )

        expanded_mask = valid_mask[..., None, None, None].expand_as(element_loss)
        masked_loss = torch.where(
            expanded_mask,
            element_loss,
            torch.zeros((), dtype=element_loss.dtype, device=element_loss.device),
        )
        numerator = masked_loss.sum()
        denominator = expanded_mask.sum().to(element_loss.dtype)
        return numerator, denominator, numerator / denominator.clamp_min(1)

    def forward(
        self,
        student_corr,
        teacher_corr,
        student_relation_gain,
        teacher_relation_gain,
        valid_mask,
    ):
        return self.forward_parts(
            student_corr,
            teacher_corr,
            student_relation_gain,
            teacher_relation_gain,
            valid_mask,
        )[2]
