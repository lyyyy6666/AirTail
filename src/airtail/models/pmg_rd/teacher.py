import math

import torch
import torch.nn as nn

from airtail.models.backbone import QuantileBackbone

from .corr import CrossKnowledgeCorrelation
from .guider import RegionScaleRelationGuider
from .state_conditioned_evidence_encoder import StateConditionedEvidenceEncoder


class TeacherRelationGenerator(nn.Module):
    """Generate the teacher relation matrix from two discovery memories."""

    def __init__(
        self,
        d_model,
        num_heads,
        num_layers,
        dim_feedforward,
        dropout,
        activation,
        num_tokens=12,
        group_chunk_size=None,
        relation_encoder_type="scpee",
        relation_query_source="state",
    ):
        super().__init__()
        dimensions = {
            "d_model": d_model,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "dim_feedforward": dim_feedforward,
        }
        for name, value in dimensions.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be a positive integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if isinstance(dropout, bool) or not isinstance(dropout, (int, float)):
            raise TypeError("dropout must be a number")
        dropout = float(dropout)
        if not math.isfinite(dropout) or not 0.0 <= dropout <= 1.0:
            raise ValueError("dropout must be between 0 and 1")
        if not isinstance(activation, str):
            raise TypeError("activation must be a string")
        if activation not in {"relu", "gelu"}:
            raise ValueError("activation must be relu or gelu")
        if isinstance(num_tokens, bool) or not isinstance(num_tokens, int):
            raise TypeError("num_tokens must be the integer 12")
        if num_tokens != 12:
            raise ValueError("num_tokens must be 12 for Relation-KD V2")
        if isinstance(group_chunk_size, bool) or (
            group_chunk_size is not None and not isinstance(group_chunk_size, int)
        ):
            raise TypeError("group_chunk_size must be None or a positive integer")
        if group_chunk_size is not None and group_chunk_size <= 0:
            raise ValueError("group_chunk_size must be positive")
        if relation_encoder_type != "scpee":
            raise ValueError("relation_encoder_type must be scpee")
        if relation_query_source not in {"state", "learned"}:
            raise ValueError("relation_query_source must be state or learned")

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        self.num_tokens = num_tokens
        self.group_chunk_size = group_chunk_size
        self.relation_encoder_type = relation_encoder_type
        self.relation_query_source = relation_query_source

        self.temporal_evidence_encoder = StateConditionedEvidenceEncoder(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            num_tokens=num_tokens,
            query_source=relation_query_source,
        )
        self.variable_evidence_encoder = StateConditionedEvidenceEncoder(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            num_tokens=num_tokens,
            query_source=relation_query_source,
        )
        self.temporal_guider = RegionScaleRelationGuider(
            d_model=d_model,
            num_heads=num_heads,
            num_tokens=num_tokens,
        )
        self.variable_guider = RegionScaleRelationGuider(
            d_model=d_model,
            num_heads=num_heads,
            num_tokens=num_tokens,
        )
        self.correlation = CrossKnowledgeCorrelation(
            num_heads=num_heads,
            num_tokens=num_tokens,
        )

    def _validate_inputs(
        self,
        relation_states,
        temporal_knowledge_memory,
        variable_knowledge_memory,
        temporal_knowledge_mask,
        variable_knowledge_mask,
    ):
        tensors = {
            "relation_states": relation_states,
            "temporal_knowledge_memory": temporal_knowledge_memory,
            "variable_knowledge_memory": variable_knowledge_memory,
            "temporal_knowledge_mask": temporal_knowledge_mask,
            "variable_knowledge_mask": variable_knowledge_mask,
        }
        for name, value in tensors.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a Tensor")

        if relation_states.ndim != 3 or relation_states.shape[1:] != (
            self.num_tokens,
            self.d_model,
        ):
            raise ValueError("relation_states must have shape [G,12,D]")
        if temporal_knowledge_memory.ndim != 3:
            raise ValueError("temporal_knowledge_memory must have shape [G,T_t,D]")
        if variable_knowledge_memory.ndim != 3:
            raise ValueError("variable_knowledge_memory must have shape [G,T_v,D]")

        groups = relation_states.shape[0]
        if groups == 0:
            raise ValueError("G must be positive")
        if temporal_knowledge_memory.shape[0] != groups:
            raise ValueError("temporal_knowledge_memory must share G with relation_states")
        if variable_knowledge_memory.shape[0] != groups:
            raise ValueError("variable_knowledge_memory must share G with relation_states")
        if temporal_knowledge_memory.shape[1] == 0:
            raise ValueError("temporal knowledge memory length T_t must be positive")
        if variable_knowledge_memory.shape[1] == 0:
            raise ValueError("variable knowledge memory length T_v must be positive")
        if temporal_knowledge_memory.shape[2] != self.d_model:
            raise ValueError("temporal_knowledge_memory last dimension must equal d_model")
        if variable_knowledge_memory.shape[2] != self.d_model:
            raise ValueError("variable_knowledge_memory last dimension must equal d_model")
        if temporal_knowledge_mask.shape != temporal_knowledge_memory.shape[:2]:
            raise ValueError("temporal_knowledge_mask must have shape [G,T_t]")
        if variable_knowledge_mask.shape != variable_knowledge_memory.shape[:2]:
            raise ValueError("variable_knowledge_mask must have shape [G,T_v]")
        if temporal_knowledge_mask.dtype != torch.bool:
            raise TypeError("temporal_knowledge_mask must have bool dtype")
        if variable_knowledge_mask.dtype != torch.bool:
            raise TypeError("variable_knowledge_mask must have bool dtype")

        reference = relation_states
        if not reference.is_floating_point():
            raise TypeError("relation_states must use a floating point dtype")
        for name, value in tensors.items():
            if name.endswith("_mask"):
                if value.device != reference.device:
                    raise ValueError(f"{name} and relation_states must share a device")
                continue
            if value.dtype != reference.dtype:
                raise TypeError(f"{name} and relation_states must share a dtype")
            if value.device != reference.device:
                raise ValueError(f"{name} and relation_states must share a device")
        parameter = next(self.parameters())
        if reference.dtype != parameter.dtype:
            raise TypeError("inputs dtype must match module dtype")
        if reference.device != parameter.device:
            raise ValueError("inputs device must match module device")

    def _forward_chunk(
        self,
        relation_states,
        temporal_knowledge_memory,
        variable_knowledge_memory,
        temporal_knowledge_mask,
        variable_knowledge_mask,
    ):
        temporal_tokens = self.temporal_evidence_encoder(
            relation_states,
            temporal_knowledge_memory,
            temporal_knowledge_mask,
        )
        variable_tokens = self.variable_evidence_encoder(
            relation_states,
            variable_knowledge_memory,
            variable_knowledge_mask,
        )
        temporal_scores = self.temporal_guider(relation_states, temporal_tokens)
        variable_scores = self.variable_guider(relation_states, variable_tokens)
        return self.correlation(temporal_scores, variable_scores)

    def forward(
        self,
        relation_states,
        temporal_knowledge_memory,
        variable_knowledge_memory,
        temporal_knowledge_mask,
        variable_knowledge_mask,
    ):
        self._validate_inputs(
            relation_states,
            temporal_knowledge_memory,
            variable_knowledge_memory,
            temporal_knowledge_mask,
            variable_knowledge_mask,
        )
        groups = relation_states.shape[0]
        chunk_size = self.group_chunk_size or groups
        if chunk_size >= groups:
            return self._forward_chunk(
                relation_states,
                temporal_knowledge_memory,
                variable_knowledge_memory,
                temporal_knowledge_mask,
                variable_knowledge_mask,
            )

        outputs = []
        for start in range(0, groups, chunk_size):
            end = min(start + chunk_size, groups)
            outputs.append(
                self._forward_chunk(
                    relation_states[start:end],
                    temporal_knowledge_memory[start:end],
                    variable_knowledge_memory[start:end],
                    temporal_knowledge_mask[start:end],
                    variable_knowledge_mask[start:end],
                )
            )
        return torch.cat(outputs, dim=0)


class TeacherRelationPath(nn.Module):
    """Compose teacher discoveries, relation generation, and relation attention.

    The path only owns the composition contract.  Its four components are
    supplied by the caller so that their formal dimensions and hyperparameters
    remain outside this module.
    """

    def __init__(
        self,
        temporal_discovery,
        variable_discovery,
        relation_generator,
        relation_attention,
    ):
        super().__init__()
        modules = {
            "temporal_discovery": temporal_discovery,
            "variable_discovery": variable_discovery,
            "relation_generator": relation_generator,
            "relation_attention": relation_attention,
        }
        for name, module in modules.items():
            if not isinstance(module, nn.Module):
                raise TypeError(f"{name} must be an nn.Module")

        self.temporal_discovery = temporal_discovery
        self.variable_discovery = variable_discovery
        self.relation_generator = relation_generator
        self.relation_attention = relation_attention

    def _validate_relation_states(self, relation_states):
        if not isinstance(relation_states, torch.Tensor):
            raise TypeError("relation_states must be a Tensor")
        if relation_states.ndim != 6:
            raise ValueError(
                "relation_states must have shape [B,H,N,3,4,D]"
            )
        batch, horizons, stations, regions, scales, hidden_dim = relation_states.shape
        if batch <= 0 or horizons <= 0 or stations <= 0:
            raise ValueError("B, H, and N must be positive")
        if regions != 3 or scales != 4:
            raise ValueError("relation_states must have shape [B,H,N,3,4,D]")
        if not relation_states.is_floating_point():
            raise TypeError("relation_states must use a floating point dtype")
        return batch, horizons, stations, hidden_dim

    @staticmethod
    def _module_reference(module):
        try:
            return next(module.parameters())
        except StopIteration:
            return None

    def _validate_input_alignment(
        self,
        relation_states,
        history_weather,
        future_weather,
        future_valid_mask,
        history_pm25,
        history_covariates,
    ):
        tensors = {
            "history_weather": history_weather,
            "history_pm25": history_pm25,
            "history_covariates": history_covariates,
        }
        if getattr(self.temporal_discovery, "use_future_meteorology", True):
            tensors["future_weather"] = future_weather
        for name, value in tensors.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a Tensor")
            if not value.is_floating_point():
                raise TypeError(f"{name} must use a floating point dtype")
            if value.dtype != relation_states.dtype:
                raise TypeError(f"{name} and relation_states must share a dtype")
            if value.device != relation_states.device:
                raise ValueError(f"{name} and relation_states must share a device")

        batch, horizons, stations = relation_states.shape[:3]
        if (
            history_weather.ndim != 4
            or history_weather.shape[0] != batch
            or history_weather.shape[2] != stations
        ):
            raise ValueError(
                "history_weather must match relation_states batch and station axes"
            )
        if getattr(self.temporal_discovery, "use_future_meteorology", True):
            if (
                future_weather.ndim != 4
                or future_weather.shape[:3] != (batch, horizons, stations)
            ):
                raise ValueError(
                    "future_weather must match relation_states batch, horizon, and station axes"
                )
        elif future_weather is not None or future_valid_mask is not None:
            raise ValueError(
                "history-only teacher relation path must not receive future meteorology"
            )
        if (
            history_pm25.ndim != 4
            or history_pm25.shape[0] != batch
            or history_pm25.shape[2] != stations
        ):
            raise ValueError(
                "history_pm25 must match relation_states batch and station axes"
            )
        if (
            history_covariates.ndim != 4
            or history_covariates.shape[:3] != history_pm25.shape[:3]
        ):
            raise ValueError(
                "history_covariates must match history_pm25 batch, history, and station axes"
            )

        if future_valid_mask is not None:
            if not isinstance(future_valid_mask, torch.Tensor):
                raise TypeError("future_valid_mask must be a Tensor")
            if future_valid_mask.dtype != torch.bool:
                raise TypeError("future_valid_mask must have bool dtype")
            if future_valid_mask.device != relation_states.device:
                raise ValueError(
                    "future_valid_mask and relation_states must share a device"
                )
            if future_valid_mask.shape != (batch, horizons, stations, 1):
                raise ValueError("future_valid_mask must have shape [B,H,N,1]")

        for module_name, module in (
            ("temporal_discovery", self.temporal_discovery),
            ("variable_discovery", self.variable_discovery),
            ("relation_generator", self.relation_generator),
            ("relation_attention", self.relation_attention),
        ):
            reference = self._module_reference(module)
            if reference is None:
                continue
            if reference.dtype != relation_states.dtype:
                raise TypeError(
                    f"relation_states dtype must match {module_name} parameters"
                )
            if reference.device != relation_states.device:
                raise ValueError(
                    f"relation_states device must match {module_name} parameters"
                )

    def _build_relation_targets(
        self,
        relation_states,
        history_weather,
        future_weather,
        future_valid_mask,
        history_pm25,
        history_covariates,
    ):
        batch, horizons, stations, hidden_dim = self._validate_relation_states(
            relation_states
        )
        self._validate_input_alignment(
            relation_states,
            history_weather,
            future_weather,
            future_valid_mask,
            history_pm25,
            history_covariates,
        )

        temporal_memory, temporal_mask = self.temporal_discovery(
            history_weather,
            future_weather,
            future_valid_mask,
        )
        variable_memory, variable_mask = self.variable_discovery(
            history_pm25,
            history_covariates,
        )

        groups = batch * horizons * stations
        flat_relation_states = relation_states.reshape(
            groups,
            3 * 4,
            hidden_dim,
        )
        teacher_corr_flat = self.relation_generator(
            flat_relation_states,
            temporal_memory,
            variable_memory,
            temporal_mask,
            variable_mask,
        )
        if not isinstance(teacher_corr_flat, torch.Tensor):
            raise TypeError("relation_generator must return a Tensor")
        if teacher_corr_flat.ndim != 4 or teacher_corr_flat.shape[0] != groups:
            raise ValueError("relation_generator must return [G,M,12,12]")
        if teacher_corr_flat.shape[2:] != (12, 12):
            raise ValueError("relation_generator must return [G,M,12,12]")
        teacher_corr = teacher_corr_flat.reshape(
            batch,
            horizons,
            stations,
            teacher_corr_flat.shape[1],
            12,
            12,
        )
        return (
            (batch, horizons, stations, hidden_dim),
            flat_relation_states,
            teacher_corr_flat,
            teacher_corr,
        )

    def forward(
        self,
        relation_states,
        history_weather,
        future_weather,
        future_valid_mask,
        history_pm25,
        history_covariates,
    ):
        shape, flat_relation_states, teacher_corr_flat, teacher_corr = (
            self._build_relation_targets(
                relation_states,
                history_weather,
                future_weather,
                future_valid_mask,
                history_pm25,
                history_covariates,
            )
        )
        enhanced_flat = self.relation_attention(
            flat_relation_states,
            teacher_corr_flat,
        )
        if not isinstance(enhanced_flat, torch.Tensor):
            raise TypeError("relation_attention must return a Tensor")
        if enhanced_flat.shape != flat_relation_states.shape:
            raise ValueError(
                "relation_attention must preserve the [G,12,D] state shape"
            )

        batch, horizons, stations, hidden_dim = shape
        enhanced_states = enhanced_flat.reshape(
            batch,
            horizons,
            stations,
            3,
            4,
            hidden_dim,
        )
        return enhanced_states, teacher_corr


class AirTailTeacher(nn.Module):
    """Compose the QC-MSSM backbone with the Teacher relation path."""

    def __init__(self, backbone, relation_path, history_weather_indices):
        super().__init__()
        if not isinstance(backbone, QuantileBackbone):
            raise TypeError("backbone must be a QuantileBackbone")
        if not isinstance(relation_path, TeacherRelationPath):
            raise TypeError("relation_path must be a TeacherRelationPath")
        if not isinstance(history_weather_indices, (list, tuple)):
            raise TypeError("history_weather_indices must be a list or tuple")

        indices = tuple(history_weather_indices)
        if not indices:
            raise ValueError("history_weather_indices must be non-empty")
        if any(isinstance(index, bool) or not isinstance(index, int) for index in indices):
            raise TypeError("history_weather_indices must contain integers")
        if any(index < 0 for index in indices):
            raise ValueError("history_weather_indices must be non-negative")
        if len(set(indices)) != len(indices):
            raise ValueError("history_weather_indices must be unique")

        decoder = backbone.decoder
        if decoder.num_regions != 3:
            raise ValueError("Relation-KD V2 requires R=3")
        if decoder.num_scales != 4:
            raise ValueError("Relation-KD V2 requires S=4")

        temporal = relation_path.temporal_discovery
        variable = relation_path.variable_discovery
        generator = relation_path.relation_generator
        attention = relation_path.relation_attention
        for name, module in (
            ("TemporalEvolutionDiscovery", temporal),
            ("VariableImpactDiscovery", variable),
            ("TeacherRelationGenerator", generator),
            ("RegionScaleRelationAttention", attention),
        ):
            if module.d_model != decoder.d_model:
                raise ValueError(f"{name} d_model must match the backbone")
        if temporal.num_weather_variables != len(indices):
            raise ValueError(
                "history_weather_indices must match "
                "TemporalEvolutionDiscovery num_weather_variables"
            )
        if variable.num_covariates != backbone.config["met_dim"]:
            raise ValueError(
                "VariableImpactDiscovery num_covariates must match backbone met_dim"
            )
        if temporal.history_length != variable.history_length:
            raise ValueError("Teacher discoveries must share the history length")
        if (
            temporal.prediction_length != decoder.horizon
            or variable.prediction_length != decoder.horizon
        ):
            raise ValueError("Teacher discovery horizons must match the backbone horizon")
        if generator.num_tokens != 12 or attention.num_tokens != 12:
            raise ValueError("Teacher relation modules must use exactly 12 tokens")
        if generator.num_heads != attention.num_heads:
            raise ValueError("Teacher relation generator and attention head counts must match")

        self.backbone = backbone
        self.relation_path = relation_path
        self.history_weather_indices = indices
        self.use_future_meteorology = temporal.use_future_meteorology

    def _validate_teacher_inputs(self, history_values, history_met, future_met, future_met_mask):
        if history_met is None:
            raise ValueError("AirTailTeacher requires history_met")
        if not isinstance(history_met, torch.Tensor):
            raise TypeError("history_met must be a Tensor")
        if history_met.ndim != 4:
            raise ValueError("history_met must have shape [B,L,N,F_hist]")
        if history_values.ndim != 4:
            raise ValueError("history_values must have shape [B,L,N,1]")
        if history_met.shape[:3] != history_values.shape[:3]:
            raise ValueError("history_met must align with history_values")
        history_channels = history_met.shape[-1]
        if any(index >= history_channels for index in self.history_weather_indices):
            raise ValueError("history_weather_indices are outside the history_met range")
        weather_channels = len(self.history_weather_indices)
        if (
            weather_channels
            != self.relation_path.temporal_discovery.num_weather_variables
        ):
            raise ValueError(
                "selected weather channels must match "
                "TemporalEvolutionDiscovery num_weather_variables"
            )

        if not self.use_future_meteorology:
            if future_met is not None or future_met_mask is not None:
                raise ValueError(
                    "history-only AirTailTeacher must not receive future meteorology"
                )
            return

        if future_met is None:
            raise ValueError("AirTailTeacher requires future_met")
        if not isinstance(future_met, torch.Tensor):
            raise TypeError("future_met must be a Tensor")
        if future_met.ndim != 4:
            raise ValueError("future_met must have shape [B,H,N,F_weather]")
        if future_met.shape[:3] != (
            history_values.shape[0],
            self.backbone.decoder.horizon,
            history_values.shape[2],
        ):
            raise ValueError("future_met must align with batch, horizon, and station axes")
        if future_met.shape[-1] != weather_channels:
            raise ValueError(
                "history and future weather channel counts must match"
            )
        if future_met_mask is not None:
            if not isinstance(future_met_mask, torch.Tensor):
                raise TypeError("future_met_mask must be a Tensor")
            if future_met_mask.shape != future_met.shape[:3] + (1,):
                raise ValueError("future_met_mask must have shape [B,H,N,1]")
            if future_met_mask.dtype != torch.bool:
                raise TypeError("future_met_mask must have bool dtype")

    def _forward(
        self,
        history_values,
        history_mask,
        future_met,
        history_met,
        history_wind_uv,
        history_time_features,
        future_met_mask,
        coords,
        neighbor_index,
        taus,
        return_diagnostics,
    ):
        self._validate_teacher_inputs(
            history_values,
            history_met,
            future_met,
            future_met_mask,
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
        relation_states = self.backbone.decoder.build_horizon_scale_states(
            ssm_states,
            history_mask,
        )
        history_weather = history_met[..., self.history_weather_indices]
        enhanced_states, teacher_corr = self.relation_path(
            relation_states,
            history_weather,
            future_met,
            future_met_mask,
            history_values,
            history_met,
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
        return prediction, teacher_corr

    def relation_target_forward(
        self,
        history_values,
        history_mask,
        future_met,
        history_met=None,
        history_wind_uv=None,
        history_time_features=None,
        future_met_mask=None,
        coords=None,
        neighbor_index=None,
        taus=None,
        return_diagnostics=False,
    ):
        """Return only the frozen Teacher relation target for Student KD."""
        self._validate_teacher_inputs(
            history_values,
            history_met,
            future_met,
            future_met_mask,
        )
        ssm_states, _, _ = self.backbone.encode_history(
            history_values,
            history_mask,
            history_met,
            history_wind_uv,
            history_time_features,
            coords,
            neighbor_index,
            return_diagnostics=False,
        )
        relation_states = self.backbone.decoder.build_horizon_scale_states(
            ssm_states,
            history_mask,
        )
        history_weather = history_met[..., self.history_weather_indices]
        _, _, _, teacher_corr = self.relation_path._build_relation_targets(
            relation_states,
            history_weather,
            future_met,
            future_met_mask,
            history_values,
            history_met,
        )
        return teacher_corr

    def forecast_forward(
        self,
        history_values,
        history_mask,
        future_met,
        history_met=None,
        history_wind_uv=None,
        history_time_features=None,
        future_met_mask=None,
        coords=None,
        neighbor_index=None,
        taus=None,
    ):
        prediction, _ = self._forward(
            history_values,
            history_mask,
            future_met,
            history_met,
            history_wind_uv,
            history_time_features,
            future_met_mask,
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
        future_met,
        history_met=None,
        history_wind_uv=None,
        history_time_features=None,
        future_met_mask=None,
        coords=None,
        neighbor_index=None,
        taus=None,
        return_diagnostics=False,
    ):
        return self._forward(
            history_values,
            history_mask,
            future_met,
            history_met,
            history_wind_uv,
            history_time_features,
            future_met_mask,
            coords,
            neighbor_index,
            taus,
            return_diagnostics,
        )
