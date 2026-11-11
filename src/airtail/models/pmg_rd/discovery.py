import math

import torch
import torch.nn as nn


class TemporalEvolutionDiscovery(nn.Module):
    """Discover per-variable weather evolution from history to each horizon.

    Inputs:
        history_weather: ``[B,L,N,F]``.
        future_weather: ``[B,H,N,F]``.
        future_valid_mask: optional ``[B,H,N,1]`` bool mask.

    Outputs:
        temporal_memory: ``[B*H*N,F,D]``.
        temporal_memory_mask: ``[B*H*N,F]`` bool mask.
    """

    def __init__(
        self,
        d_model,
        num_weather_variables,
        history_length,
        prediction_length,
        dropout,
        group_chunk_size=None,
        use_future_meteorology=True,
    ):
        super().__init__()
        dimensions = {
            "d_model": d_model,
            "num_weather_variables": num_weather_variables,
            "history_length": history_length,
            "prediction_length": prediction_length,
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
        if isinstance(group_chunk_size, bool) or (
            group_chunk_size is not None and not isinstance(group_chunk_size, int)
        ):
            raise TypeError("group_chunk_size must be None or a positive integer")
        if group_chunk_size is not None and group_chunk_size <= 0:
            raise ValueError("group_chunk_size must be positive")
        if not isinstance(use_future_meteorology, bool):
            raise TypeError("use_future_meteorology must be a bool")

        self.d_model = d_model
        self.num_weather_variables = num_weather_variables
        self.history_length = history_length
        self.prediction_length = prediction_length
        self.group_chunk_size = group_chunk_size
        self.use_future_meteorology = use_future_meteorology

        self.history_value_projection = nn.Linear(1, d_model)
        self.future_value_projection = nn.Linear(1, d_model)
        self.variable_embedding = nn.Parameter(
            torch.empty(num_weather_variables, d_model)
        )
        self.history_lag_embedding = nn.Parameter(
            torch.empty(history_length, d_model)
        )
        self.forecast_horizon_embedding = nn.Parameter(
            torch.empty(prediction_length, d_model)
        )

        self.query_projection = nn.Linear(d_model, d_model, bias=False)
        self.key_projection = nn.Linear(d_model, d_model, bias=False)
        self.value_projection = nn.Linear(d_model, d_model, bias=False)
        self.attention_weight_dropout = nn.Dropout(dropout)

        nn.init.normal_(self.variable_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.history_lag_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.forecast_horizon_embedding, mean=0.0, std=0.02)

    def _validate_inputs(self, history_weather, future_weather, future_valid_mask):
        if not isinstance(history_weather, torch.Tensor):
            raise TypeError("history_weather must be a Tensor")
        if (
            history_weather.ndim != 4
            or history_weather.shape[1] != self.history_length
            or history_weather.shape[-1] != self.num_weather_variables
        ):
            raise ValueError("history_weather must have shape [B,L,N,F]")
        if history_weather.shape[0] == 0 or history_weather.shape[2] == 0:
            raise ValueError("B and N must be positive")
        if not history_weather.is_floating_point():
            raise TypeError("history_weather must use a floating point dtype")

        if not self.use_future_meteorology:
            if future_weather is not None or future_valid_mask is not None:
                raise ValueError(
                    "history-only temporal discovery must not receive future meteorology"
                )
            return

        if not isinstance(future_weather, torch.Tensor):
            raise TypeError("future_weather must be a Tensor")
        if (
            future_weather.ndim != 4
            or future_weather.shape[1] != self.prediction_length
            or future_weather.shape[-1] != self.num_weather_variables
        ):
            raise ValueError("future_weather must have shape [B,H,N,F]")
        if (
            history_weather.shape[0] != future_weather.shape[0]
            or history_weather.shape[2] != future_weather.shape[2]
            or history_weather.shape[3] != future_weather.shape[3]
        ):
            raise ValueError("history_weather and future_weather must share the same B, N, and F")
        if not future_weather.is_floating_point():
            raise TypeError("future_weather must use a floating point dtype")
        if history_weather.dtype != future_weather.dtype:
            raise TypeError("history_weather and future_weather must share the same dtype")
        if history_weather.device != future_weather.device:
            raise ValueError("history_weather and future_weather must share the same device")

        if future_valid_mask is None:
            return
        if not isinstance(future_valid_mask, torch.Tensor):
            raise TypeError("future_valid_mask must be a Tensor")
        if future_valid_mask.shape != future_weather.shape[:3] + (1,):
            raise ValueError("future_valid_mask must have shape [B,H,N,1]")
        if future_valid_mask.dtype != torch.bool:
            raise TypeError("future_valid_mask must have bool dtype")
        if future_valid_mask.device != future_weather.device:
            raise ValueError("future_valid_mask and weather inputs must share the same device")

    def _forward_chunk(
        self,
        query,
        key,
        value,
        group_valid,
        start,
        end,
        stations,
    ):
        group_index = torch.arange(start, end, device=query.device)
        batch_index = group_index // (self.prediction_length * stations)
        station_index = group_index % stations

        query = query[start:end]
        key = key[batch_index, station_index]
        value = value[batch_index, station_index]

        score = torch.matmul(
            query.unsqueeze(-2), key.transpose(-2, -1)
        ).squeeze(-2)
        score = score / math.sqrt(self.d_model)
        relation = torch.softmax(score, dim=-1)
        relation = self.attention_weight_dropout(relation)
        memory = torch.matmul(relation.unsqueeze(-2), value).squeeze(-2)

        memory_mask = group_valid[start:end, None].expand(
            end - start, self.num_weather_variables
        )
        memory = memory * memory_mask[..., None].to(memory.dtype)
        return memory, memory_mask

    def forward(self, history_weather, future_weather, future_valid_mask=None):
        self._validate_inputs(history_weather, future_weather, future_valid_mask)
        batch, _, stations, _ = history_weather.shape
        groups = batch * self.prediction_length * stations

        history_tokens = self.history_value_projection(history_weather.unsqueeze(-1))
        history_tokens = (
            history_tokens
            + self.variable_embedding[None, None, None]
            + self.history_lag_embedding[None, :, None, None]
        )
        if self.use_future_meteorology:
            future_tokens = self.future_value_projection(future_weather.unsqueeze(-1))
            future_tokens = (
                future_tokens
                + self.variable_embedding[None, None, None]
                + self.forecast_horizon_embedding[None, :, None, None]
            )
        else:
            future_tokens = (
                self.variable_embedding[None, None, None]
                + self.forecast_horizon_embedding[None, :, None, None]
            ).expand(batch, -1, stations, -1, -1)

        query = self.query_projection(future_tokens).reshape(
            groups, self.num_weather_variables, self.d_model
        )
        key = self.key_projection(history_tokens).permute(0, 2, 3, 1, 4)
        value = self.value_projection(history_tokens).permute(0, 2, 3, 1, 4)

        if future_valid_mask is None:
            group_valid = torch.ones(groups, dtype=torch.bool, device=query.device)
        else:
            group_valid = future_valid_mask.reshape(groups)

        chunk_size = self.group_chunk_size or groups
        memory_chunks = []
        mask_chunks = []
        for start in range(0, groups, chunk_size):
            end = min(start + chunk_size, groups)
            memory, memory_mask = self._forward_chunk(
                query,
                key,
                value,
                group_valid,
                start,
                end,
                stations,
            )
            memory_chunks.append(memory)
            mask_chunks.append(memory_mask)

        return torch.cat(memory_chunks, dim=0), torch.cat(mask_chunks, dim=0)


class VariableImpactDiscovery(nn.Module):
    """Discover horizon-specific covariate impact from history-only inputs.

    Inputs:
        history_pm25: ``[B,L,N,1]``.
        history_covariates: ``[B,L,N,Fv]``.

    Outputs:
        variable_memory: ``[B*H*N,Fv,D]``.
        variable_memory_mask: ``[B*H*N,Fv]`` bool mask.
    """

    def __init__(
        self,
        d_model,
        num_covariates,
        history_length,
        prediction_length,
        dropout,
        group_chunk_size=None,
    ):
        super().__init__()
        dimensions = {
            "d_model": d_model,
            "num_covariates": num_covariates,
            "history_length": history_length,
            "prediction_length": prediction_length,
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
        if isinstance(group_chunk_size, bool) or (
            group_chunk_size is not None
            and not isinstance(group_chunk_size, int)
        ):
            raise TypeError("group_chunk_size must be None or a positive integer")
        if group_chunk_size is not None and group_chunk_size <= 0:
            raise ValueError("group_chunk_size must be positive")

        self.d_model = d_model
        self.num_covariates = num_covariates
        self.history_length = history_length
        self.prediction_length = prediction_length
        self.group_chunk_size = group_chunk_size

        self.target_state_projection = nn.Linear(2, d_model)
        self.forecast_horizon_embedding = nn.Parameter(
            torch.empty(prediction_length, d_model)
        )
        self.target_query_projection = nn.Linear(d_model, d_model)
        self.target_key_projection = nn.Linear(d_model, d_model)
        self.target_value_projection = nn.Linear(d_model, d_model)
        self.target_attention_dropout = nn.Dropout(dropout)

        self.covariate_value_projection = nn.Linear(1, d_model)
        self.variable_embedding = nn.Parameter(
            torch.empty(num_covariates, d_model)
        )
        self.history_lag_embedding = nn.Parameter(
            torch.empty(history_length, d_model)
        )
        self.impact_query_projection = nn.Linear(d_model, d_model)
        self.impact_key_projection = nn.Linear(d_model, d_model)
        self.impact_value_projection = nn.Linear(d_model, d_model)
        self.impact_attention_dropout = nn.Dropout(dropout)

        nn.init.normal_(self.forecast_horizon_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.variable_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.history_lag_embedding, mean=0.0, std=0.02)

    def _validate_inputs(self, history_pm25, history_covariates):
        if not isinstance(history_pm25, torch.Tensor):
            raise TypeError("history_pm25 must be a Tensor")
        if not isinstance(history_covariates, torch.Tensor):
            raise TypeError("history_covariates must be a Tensor")
        if (
            history_pm25.ndim != 4
            or history_pm25.shape[1] != self.history_length
            or history_pm25.shape[-1] != 1
        ):
            raise ValueError("history_pm25 must have shape [B,L,N,1]")
        if (
            history_covariates.ndim != 4
            or history_covariates.shape[1] != self.history_length
            or history_covariates.shape[-1] != self.num_covariates
        ):
            raise ValueError("history_covariates must have shape [B,L,N,Fv]")
        if history_pm25.shape[:3] != history_covariates.shape[:3]:
            raise ValueError(
                "history_pm25 and history_covariates must share the same B, L, and N"
            )
        if history_pm25.shape[0] == 0 or history_pm25.shape[2] == 0:
            raise ValueError("B and N must be positive")
        if (
            not history_pm25.is_floating_point()
            or not history_covariates.is_floating_point()
        ):
            raise TypeError(
                "history_pm25 and history_covariates must use floating point dtypes"
            )
        if history_pm25.dtype != history_covariates.dtype:
            raise TypeError(
                "history_pm25 and history_covariates must share the same dtype"
            )
        if history_pm25.device != history_covariates.device:
            raise ValueError(
                "history_pm25 and history_covariates must share the same device"
            )

    def _forward_chunk(
        self,
        horizon_query,
        target_key,
        target_value,
        impact_key,
        impact_value,
        start,
        end,
        stations,
    ):
        group_index = torch.arange(start, end, device=horizon_query.device)
        batch_index = group_index // (self.prediction_length * stations)
        horizon_index = (group_index // stations) % self.prediction_length
        station_index = group_index % stations

        query = horizon_query[horizon_index]
        key = target_key[batch_index, station_index]
        value = target_value[batch_index, station_index]

        target_score = torch.matmul(
            query.unsqueeze(-2), key.transpose(-2, -1)
        ).squeeze(-2)
        target_score = target_score / math.sqrt(self.d_model)
        target_relation = torch.softmax(target_score, dim=-1)
        target_relation = self.target_attention_dropout(target_relation)
        pm25_dynamics = torch.matmul(
            target_relation.unsqueeze(-2), value
        ).squeeze(-2)

        query = self.impact_query_projection(
            pm25_dynamics[:, None] + self.variable_embedding[None]
        )
        key = impact_key[batch_index, station_index]
        value = impact_value[batch_index, station_index]

        impact_score = torch.matmul(
            query.unsqueeze(-2), key.transpose(-2, -1)
        ).squeeze(-2)
        impact_score = impact_score / math.sqrt(self.d_model)
        relation = torch.softmax(impact_score, dim=-1)
        relation = self.impact_attention_dropout(relation)
        return torch.matmul(relation.unsqueeze(-2), value).squeeze(-2)

    def forward(self, history_pm25, history_covariates):
        self._validate_inputs(history_pm25, history_covariates)
        batch, _, stations, _ = history_pm25.shape
        groups = batch * self.prediction_length * stations

        delta_pm25 = torch.zeros_like(history_pm25)
        delta_pm25[:, 1:] = history_pm25[:, 1:] - history_pm25[:, :-1]
        target_state = torch.cat((history_pm25, delta_pm25), dim=-1)
        target_token = self.target_state_projection(target_state)
        target_token = (
            target_token + self.history_lag_embedding[None, :, None]
        )

        horizon_query = self.target_query_projection(
            self.forecast_horizon_embedding
        )
        target_key = self.target_key_projection(target_token).permute(
            0, 2, 1, 3
        )
        target_value = self.target_value_projection(target_token).permute(
            0, 2, 1, 3
        )

        covariate_token = self.covariate_value_projection(
            history_covariates.unsqueeze(-1)
        )
        covariate_token = (
            covariate_token
            + self.variable_embedding[None, None, None]
            + self.history_lag_embedding[None, :, None, None]
        )
        impact_key = self.impact_key_projection(covariate_token).permute(
            0, 2, 3, 1, 4
        )
        impact_value = self.impact_value_projection(covariate_token).permute(
            0, 2, 3, 1, 4
        )

        chunk_size = self.group_chunk_size or groups
        memory_chunks = []
        for start in range(0, groups, chunk_size):
            end = min(start + chunk_size, groups)
            memory_chunks.append(
                self._forward_chunk(
                    horizon_query,
                    target_key,
                    target_value,
                    impact_key,
                    impact_value,
                    start,
                    end,
                    stations,
                )
            )

        variable_memory = torch.cat(memory_chunks, dim=0)
        variable_memory_mask = torch.ones(
            groups,
            self.num_covariates,
            dtype=torch.bool,
            device=history_pm25.device,
        )
        return variable_memory, variable_memory_mask
