import torch
import torch.nn as nn

from airtail.models.temporal_stem import CausalMultiscaleTemporalStem
from airtail.models.contracts import QuantileModelOutput
from airtail.models.spatial_mixer import (
    HistoryConditionedDirectedSpatialMixer,
)
from airtail.models.horizon_decoder import FixedCapacityHorizonDecoder
from airtail.models.sa_evt import UnifiedQuantileHead
from airtail.models.qcmm import (
    QuantileConditionedMultiscaleMambaBlock,
)


class QuantileBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        channels = config["hidden_dim"]
        kernels = config["temporal"]["kernels"]
        self.value_proj = nn.Linear(config["input_dim"], channels)
        self.met_proj = (
            nn.Linear(config["met_dim"], channels, bias=False)
            if config["met_dim"]
            else None
        )
        self.time_proj = (
            nn.Linear(config["time_dim"], channels, bias=False)
            if config["time_dim"]
            else None
        )
        self.input_norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(config["dropout"])
        self.temporal = CausalMultiscaleTemporalStem(channels, kernels)
        self.spatial_mode = config["spatial"]["mode"]
        if self.spatial_mode == "history_directed":
            spatial = config["spatial"]
            self.spatial = HistoryConditionedDirectedSpatialMixer(
                channels,
                config["num_nodes"],
                config["node_emb_dim"],
                geo=spatial["geo"],
                wind=spatial["wind"],
                content=spatial["content"],
                scale_conditioned=spatial["scale_conditioned"],
                node_adaptive=spatial["node_adaptive"],
                geo_length_scale=spatial["geo_length_scale"],
                wind_length_scale=spatial["wind_length_scale"],
                beta_geo=spatial["beta_geo"],
                beta_wind=spatial["beta_wind"],
                beta_adaptive=spatial["beta_adaptive"],
                chunk_size=spatial["chunk_size"],
            )
        else:
            self.spatial = None
        self.ssm = QuantileConditionedMultiscaleMambaBlock(
            channels,
            config["ssm"]["d_state"],
            len(kernels),
            config["ssm"]["delta_min"],
            config["ssm"]["delta_max"],
            delta_mode=config["ssm"]["delta_mode"],
            history_gamma=config["ssm"]["history_gamma"],
            backend=config["ssm"]["backend"],
        )
        self.decoder = FixedCapacityHorizonDecoder(
            channels,
            len(kernels),
            config["horizon"],
            scale_fusion=config["decoder"]["scale_fusion"],
        )
        self.head = UnifiedQuantileHead(channels, config["head"])

    def _validate_inputs(
        self,
        history_values,
        history_mask,
        history_met,
        history_wind_uv,
        history_time_features,
        coords,
        neighbor_index,
    ):
        if history_values.ndim != 4 or history_values.shape[-1] != self.config["input_dim"]:
            raise ValueError("history_values must have shape [B,L,N,Fx]")
        if history_mask.ndim != 4 or history_mask.shape[:3] != history_values.shape[:3]:
            raise ValueError("history_mask must have shape [B,L,N,1] or [B,L,N,Fx]")
        if history_mask.shape[-1] not in {1, self.config["input_dim"]}:
            raise ValueError("history_mask feature dimension must be 1 or Fx")
        if history_met is not None and (
            history_met.ndim != 4
            or history_met.shape[:3] != history_values.shape[:3]
            or history_met.shape[-1] != self.config["met_dim"]
        ):
            raise ValueError("history_met must have shape [B,L,N,Fm]")
        if history_wind_uv is not None and history_wind_uv.shape != (
            *history_values.shape[:3],
            2,
        ):
            raise ValueError("history_wind_uv must have shape [B,L,N,2]")
        if history_time_features is not None and (
            history_time_features.ndim != 3
            or history_time_features.shape[:2] != history_values.shape[:2]
            or history_time_features.shape[-1] != self.config["time_dim"]
        ):
            raise ValueError("history_time_features must have shape [B,L,Ft]")
        if self.spatial_mode == "history_directed" and neighbor_index is None:
            raise ValueError("history_directed spatial mode requires neighbor_index")
        if (
            self.spatial_mode == "history_directed"
            and self.config["spatial"]["wind"]
            and history_wind_uv is None
        ):
            raise ValueError("wind spatial edges require history_wind_uv")
        if self.spatial_mode == "history_directed" and (
            self.config["spatial"]["geo"] or self.config["spatial"]["wind"]
        ) and coords is None:
            raise ValueError("enabled geo or wind spatial edges require coords")

    def _input_states(self, history_values, history_mask, history_met, history_time_features):
        feature_mask = history_mask.to(dtype=history_values.dtype)
        if feature_mask.shape[-1] == 1:
            feature_mask = feature_mask.expand_as(history_values)
        states = self.value_proj(history_values * feature_mask)
        if history_met is not None:
            states = states + self.met_proj(history_met)
        if history_time_features is not None:
            states = states + self.time_proj(history_time_features)[:, :, None]
        return self.dropout(self.input_norm(states))

    def encode_history(
        self,
        history_values,
        history_mask,
        history_met=None,
        history_wind_uv=None,
        history_time_features=None,
        coords=None,
        neighbor_index=None,
        return_diagnostics=False,
    ):
        self._validate_inputs(
            history_values,
            history_mask,
            history_met,
            history_wind_uv,
            history_time_features,
            coords,
            neighbor_index,
        )
        states = self._input_states(
            history_values, history_mask, history_met, history_time_features
        )
        scale_states = self.temporal(states)
        if self.spatial_mode == "history_directed":
            scale_states, spatial_summary, spatial_diag = self.spatial(
                scale_states,
                history_wind_uv,
                coords,
                neighbor_index,
                return_diagnostics=return_diagnostics,
            )
        else:
            spatial_summary = {
                "sum_mean": scale_states.new_tensor(1.0),
                "entropy_mean": scale_states.new_tensor(0.0),
                "self_weight_mean": scale_states.new_tensor(1.0),
            }
            spatial_diag = None
        ssm_states, delta_summary, half_life_summary, ssm_diag = self.ssm(
            scale_states, history_mask, return_diagnostics=return_diagnostics
        )
        diagnostics = None
        if return_diagnostics:
            diagnostics = {
                "temporal_scale_states": scale_states,
                "ssm_states": ssm_states,
                **(spatial_diag or {}),
                **(ssm_diag or {}),
            }
        summaries = (delta_summary, half_life_summary, spatial_summary)
        return ssm_states, summaries, diagnostics

    def encode(
        self,
        history_values,
        history_mask,
        history_met=None,
        history_wind_uv=None,
        history_time_features=None,
        coords=None,
        neighbor_index=None,
        return_diagnostics=False,
    ):
        ssm_states, summaries, history_diagnostics = self.encode_history(
            history_values,
            history_mask,
            history_met,
            history_wind_uv,
            history_time_features,
            coords,
            neighbor_index,
            return_diagnostics=return_diagnostics,
        )
        horizon_scale_states = self.decoder.build_horizon_scale_states(
            ssm_states, history_mask
        )
        horizon_states, scale_weights, horizon_scale_states = (
            self.decoder.aggregate_scales(horizon_scale_states, ssm_states)
        )
        diagnostics = None
        if return_diagnostics:
            diagnostics = {
                "temporal_scale_states": history_diagnostics[
                    "temporal_scale_states"
                ],
                "ssm_states": ssm_states,
                "horizon_scale_states": horizon_scale_states,
                **{
                    key: value
                    for key, value in history_diagnostics.items()
                    if key not in {"temporal_scale_states", "ssm_states"}
                },
            }
        return horizon_states, scale_weights, summaries, diagnostics

    def predict_from_horizon(
        self, horizon_states, scale_weights, summaries, taus, diagnostics=None
    ):
        head = self.head(horizon_states, taus)
        delta_summary, half_life_summary, spatial_summary = summaries
        return QuantileModelOutput(
            quantiles=head.quantiles,
            median=head.median,
            bulk_coefficients=head.bulk_coefficients,
            threshold_u=head.threshold_u,
            threshold_slope=head.threshold_slope,
            tail_sigma=head.tail_sigma,
            tail_xi=head.tail_xi,
            scale_weights=scale_weights,
            delta_summary=delta_summary,
            half_life_summary=half_life_summary,
            spatial_weight_summary=spatial_summary,
            diagnostics=diagnostics,
        )

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
        encoded = self.encode(
            history_values,
            history_mask,
            history_met,
            history_wind_uv,
            history_time_features,
            coords,
            neighbor_index,
            return_diagnostics=return_diagnostics,
        )
        return self.predict_from_horizon(*encoded[:3], taus, diagnostics=encoded[3])
