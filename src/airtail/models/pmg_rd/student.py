import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from airtail.models.backbone import QuantileBackbone


class StudentRelationPredictor(nn.Module):
    """Predict per-head low-rank relations from ``[G,12,D]`` states."""

    def __init__(
        self,
        d_model,
        num_heads,
        relation_rank,
        mlp_hidden_dim,
        dropout,
        activation,
        num_tokens=12,
        group_chunk_size=None,
    ):
        super().__init__()
        dimensions = {
            "d_model": d_model,
            "num_heads": num_heads,
            "relation_rank": relation_rank,
            "mlp_hidden_dim": mlp_hidden_dim,
        }
        for name, value in dimensions.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be a positive integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if isinstance(dropout, bool) or not isinstance(dropout, (int, float)):
            raise TypeError("dropout must be a number")
        dropout = float(dropout)
        if not math.isfinite(dropout) or not 0.0 <= dropout <= 1.0:
            raise ValueError("dropout must be between 0 and 1")
        if not isinstance(activation, str):
            raise TypeError("activation must be a string")
        if activation not in {"relu", "gelu", "silu"}:
            raise ValueError("activation must be relu, gelu, or silu")
        if isinstance(num_tokens, bool) or not isinstance(num_tokens, int):
            raise TypeError("num_tokens must be the integer 12")
        if num_tokens != 12:
            raise ValueError("num_tokens must be 12 for Relation-KD V2")
        if isinstance(group_chunk_size, bool) or (
            group_chunk_size is not None
            and not isinstance(group_chunk_size, int)
        ):
            raise TypeError("group_chunk_size must be None or a positive integer")
        if group_chunk_size is not None and group_chunk_size <= 0:
            raise ValueError("group_chunk_size must be positive")

        self.d_model = d_model
        self.num_heads = num_heads
        self.relation_rank = relation_rank
        self.mlp_hidden_dim = mlp_hidden_dim
        self.num_tokens = num_tokens
        self.activation_name = activation
        self.group_chunk_size = group_chunk_size

        factor_dim = num_heads * relation_rank
        self.query_hidden = nn.Linear(d_model, mlp_hidden_dim, bias=True)
        self.query_dropout = nn.Dropout(dropout)
        self.query_output = nn.Linear(mlp_hidden_dim, factor_dim, bias=True)

        self.key_hidden = nn.Linear(d_model, mlp_hidden_dim, bias=True)
        self.key_dropout = nn.Dropout(dropout)
        self.key_output = nn.Linear(mlp_hidden_dim, factor_dim, bias=True)

    def _activation(self, hidden_states):
        if self.activation_name == "relu":
            return F.relu(hidden_states)
        if self.activation_name == "gelu":
            return F.gelu(hidden_states)
        return F.silu(hidden_states)

    def _forward_chunk(self, student_states):
        groups = student_states.shape[0]

        query_factor = self.query_output(
            self.query_dropout(
                self._activation(self.query_hidden(student_states))
            )
        )
        key_factor = self.key_output(
            self.key_dropout(
                self._activation(self.key_hidden(student_states))
            )
        )

        query_factor = query_factor.reshape(
            groups,
            self.num_tokens,
            self.num_heads,
            self.relation_rank,
        ).transpose(1, 2)
        key_factor = key_factor.reshape(
            groups,
            self.num_tokens,
            self.num_heads,
            self.relation_rank,
        ).transpose(1, 2)

        return torch.matmul(
            query_factor,
            key_factor.transpose(-2, -1),
        ) / math.sqrt(self.relation_rank)

    def forward(self, student_states):
        if not isinstance(student_states, torch.Tensor):
            raise TypeError("student_states must be a Tensor")
        if student_states.ndim != 3:
            raise ValueError("student_states must be a rank-3 Tensor [G,12,D]")

        groups, num_tokens, hidden_dim = student_states.shape
        if groups == 0:
            raise ValueError("G must be positive")
        if num_tokens != self.num_tokens:
            raise ValueError("student_states must contain exactly 12 tokens")
        if hidden_dim != self.d_model:
            raise ValueError("student_states last dimension must equal d_model")
        if not student_states.is_floating_point():
            raise TypeError("student_states must use a floating point dtype")
        if student_states.dtype != self.query_hidden.weight.dtype:
            raise TypeError("student_states dtype must match module dtype")
        if student_states.device != self.query_hidden.weight.device:
            raise ValueError("student_states device must match module device")

        if self.group_chunk_size is None or self.group_chunk_size >= groups:
            return self._forward_chunk(student_states)

        output_chunks = []
        for start in range(0, groups, self.group_chunk_size):
            end = min(start + self.group_chunk_size, groups)
            output_chunks.append(self._forward_chunk(student_states[start:end]))
        return torch.cat(output_chunks, dim=0)


class StudentRelationPath(nn.Module):
    """Compose relation prediction and region-scale enhancement."""

    def __init__(self, relation_predictor, relation_attention):
        super().__init__()
        modules = {
            "relation_predictor": relation_predictor,
            "relation_attention": relation_attention,
        }
        for name, module in modules.items():
            if not isinstance(module, nn.Module):
                raise TypeError(f"{name} must be an nn.Module")

        self.relation_predictor = relation_predictor
        self.relation_attention = relation_attention

    @staticmethod
    def _module_reference(module):
        try:
            return next(module.parameters())
        except StopIteration:
            return None

    def _validate_relation_states(self, relation_states):
        if not isinstance(relation_states, torch.Tensor):
            raise TypeError("relation_states must be a Tensor")
        if relation_states.ndim != 6:
            raise ValueError("relation_states must have shape [B,H,N,3,4,D]")

        batch, horizons, stations, regions, scales, hidden_dim = relation_states.shape
        if batch <= 0 or horizons <= 0 or stations <= 0 or hidden_dim <= 0:
            raise ValueError("B, H, N, and D must be positive")
        if regions != 3 or scales != 4:
            raise ValueError("relation_states must have shape [B,H,N,3,4,D]")
        if not relation_states.is_floating_point():
            raise TypeError("relation_states must use a floating point dtype")

        for module_name, module in (
            ("relation_predictor", self.relation_predictor),
            ("relation_attention", self.relation_attention),
        ):
            reference = self._module_reference(module)
            if reference is None:
                continue
            if relation_states.dtype != reference.dtype:
                raise TypeError(
                    f"relation_states dtype must match {module_name} parameters"
                )
            if relation_states.device != reference.device:
                raise ValueError(
                    f"relation_states device must match {module_name} parameters"
                )

        return batch, horizons, stations, hidden_dim

    def _forward(self, relation_states, apply_attention):
        batch, horizons, stations, hidden_dim = self._validate_relation_states(
            relation_states
        )
        groups = batch * horizons * stations
        flat_relation_states = relation_states.reshape(groups, 12, hidden_dim)

        student_corr_flat = self.relation_predictor(flat_relation_states)
        if not isinstance(student_corr_flat, torch.Tensor):
            raise TypeError("relation_predictor must return a Tensor")
        if (
            student_corr_flat.ndim != 4
            or student_corr_flat.shape[0] != groups
            or student_corr_flat.shape[1] <= 0
            or student_corr_flat.shape[2:] != (12, 12)
        ):
            raise ValueError("relation_predictor must return [G,M,12,12]")
        if student_corr_flat.dtype != relation_states.dtype:
            raise TypeError(
                "relation_predictor output and relation_states must share a dtype"
            )
        if student_corr_flat.device != relation_states.device:
            raise ValueError(
                "relation_predictor output and relation_states must share a device"
            )

        if apply_attention:
            enhanced_flat = self.relation_attention(
                flat_relation_states,
                student_corr_flat,
            )
            if not isinstance(enhanced_flat, torch.Tensor):
                raise TypeError("relation_attention must return a Tensor")
            if enhanced_flat.shape != flat_relation_states.shape:
                raise ValueError(
                    "relation_attention must preserve the [G,12,D] state shape"
                )
            if enhanced_flat.dtype != relation_states.dtype:
                raise TypeError(
                    "relation_attention output and relation_states must share a dtype"
                )
            if enhanced_flat.device != relation_states.device:
                raise ValueError(
                    "relation_attention output and relation_states must share a device"
                )
        else:
            enhanced_flat = flat_relation_states

        enhanced_states = enhanced_flat.reshape(
            batch,
            horizons,
            stations,
            3,
            4,
            hidden_dim,
        )
        student_corr = student_corr_flat.reshape(
            batch,
            horizons,
            stations,
            student_corr_flat.shape[1],
            12,
            12,
        )
        return enhanced_states, student_corr

    def forward(self, relation_states):
        return self._forward(relation_states, apply_attention=True)

    def predict_without_attention(self, relation_states):
        """Predict relations while bypassing forecast enhancement."""
        return self._forward(relation_states, apply_attention=False)


class AirTailStudent(nn.Module):
    """Compose the QC-MSSM backbone with the Student relation path."""

    def __init__(
        self,
        backbone,
        relation_path,
        history_met_indices=None,
        relation_attention_enabled=True,
    ):
        super().__init__()
        if not isinstance(backbone, QuantileBackbone):
            raise TypeError("backbone must be a QuantileBackbone")
        if not isinstance(relation_path, StudentRelationPath):
            raise TypeError("relation_path must be a StudentRelationPath")
        if history_met_indices is None:
            history_met_indices = tuple(range(backbone.config["met_dim"]))
        if not isinstance(history_met_indices, (list, tuple)):
            raise TypeError("history_met_indices must be a list or tuple")
        if not isinstance(relation_attention_enabled, bool):
            raise TypeError("relation_attention_enabled must be a bool")
        indices = tuple(history_met_indices)
        if any(isinstance(index, bool) or not isinstance(index, int) for index in indices):
            raise TypeError("history_met_indices must contain integers")
        if any(index < 0 for index in indices) or len(set(indices)) != len(indices):
            raise ValueError("history_met_indices must be unique and non-negative")
        if len(indices) != backbone.config["met_dim"]:
            raise ValueError("history_met_indices must match Student met_dim")

        self.backbone = backbone
        self.relation_path = relation_path
        self.history_met_indices = indices
        self.relation_attention_enabled = relation_attention_enabled

    def _student_inputs(self, history_met, history_time_features):
        if self.history_met_indices:
            if history_met is None:
                raise ValueError("configured Student history variables require history_met")
            if max(self.history_met_indices) >= history_met.shape[-1]:
                raise ValueError("Student history_met_indices exceed input channels")
            history_met = history_met[..., self.history_met_indices]
        else:
            history_met = None
        if self.backbone.config["time_dim"] == 0:
            history_time_features = None
        return history_met, history_time_features

    def _forward(
        self,
        history_values,
        history_mask,
        history_met,
        history_wind_uv,
        history_time_features,
        coords,
        neighbor_index,
        taus,
        return_diagnostics,
    ):
        history_met, history_time_features = self._student_inputs(
            history_met, history_time_features
        )
        ssm_states, summaries, history_diagnostics = self.backbone.encode_history(
            history_values,
            history_mask,
            history_met,
            history_wind_uv,
            history_time_features,
            coords,
            neighbor_index,
            return_diagnostics=return_diagnostics,
        )
        horizon_scale_states = self.backbone.decoder.build_horizon_scale_states(
            ssm_states,
            history_mask,
        )
        if self.relation_attention_enabled:
            enhanced_states, student_corr = self.relation_path(
                horizon_scale_states
            )
        else:
            enhanced_states, student_corr = self.relation_path.predict_without_attention(
                horizon_scale_states
            )
        horizon_states, scale_weights, _ = self.backbone.decoder.aggregate_scales(
            enhanced_states,
            ssm_states,
        )

        diagnostics = None
        if return_diagnostics:
            diagnostics = {
                "temporal_scale_states": history_diagnostics[
                    "temporal_scale_states"
                ],
                "ssm_states": ssm_states,
                "horizon_scale_states": enhanced_states,
                **{
                    key: value
                    for key, value in history_diagnostics.items()
                    if key not in {"temporal_scale_states", "ssm_states"}
                },
            }

        prediction = self.backbone.predict_from_horizon(
            horizon_states,
            scale_weights,
            summaries,
            taus,
            diagnostics=diagnostics,
        )
        return prediction, student_corr

    def deploy_forward(
        self,
        history_values,
        history_mask,
        history_met=None,
        history_wind_uv=None,
        history_time_features=None,
        coords=None,
        neighbor_index=None,
        taus=None,
    ):
        prediction, _ = self._forward(
            history_values,
            history_mask,
            history_met,
            history_wind_uv,
            history_time_features,
            coords,
            neighbor_index,
            taus,
            False,
        )
        return prediction

    def forward(
        self,
        history_values,
        history_mask,
        history_met=None,
        history_wind_uv=None,
        history_time_features=None,
        coords=None,
        neighbor_index=None,
        taus=None,
        return_diagnostics=False,
    ):
        return self._forward(
            history_values,
            history_mask,
            history_met,
            history_wind_uv,
            history_time_features,
            coords,
            neighbor_index,
            taus,
            return_diagnostics,
        )
