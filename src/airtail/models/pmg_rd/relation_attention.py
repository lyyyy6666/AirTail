import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RegionScaleRelationAttention(nn.Module):
    """用逐头 Corr 增强每个 group 内的 12 个区域–尺度状态。

    输入：
        region_scale_states: ``[G,12,D]``。
        corr: ``[G,M,12,12]``。

    输出：
        relation_states: ``[G,12,D]``。
    """

    def __init__(
        self,
        d_model,
        num_heads,
        dim_feedforward,
        dropout,
        activation,
        num_tokens=12,
        group_chunk_size=None,
        relation_gain_init=1.0,
        block_gain_init=0.0,
    ):
        super().__init__()
        if d_model <= 0 or num_heads <= 0:
            raise ValueError("d_model and num_heads must be positive")
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if dim_feedforward <= 0:
            raise ValueError("dim_feedforward must be positive")
        if not 0.0 <= dropout <= 1.0:
            raise ValueError("dropout must be between 0 and 1")
        if activation not in {"relu", "gelu", "silu"}:
            raise ValueError("activation must be relu, gelu, or silu")
        if num_tokens != 12:
            raise ValueError("num_tokens must be 12 for Relation-KD V2")
        if isinstance(group_chunk_size, bool) or (
            group_chunk_size is not None and not isinstance(group_chunk_size, int)
        ):
            raise TypeError("group_chunk_size must be None or a positive integer")
        if group_chunk_size is not None and group_chunk_size <= 0:
            raise ValueError("group_chunk_size must be positive")

        gain_initializers = (
            ("relation_gain_init", relation_gain_init),
            ("block_gain_init", block_gain_init),
        )
        validated_gains = {}
        for name, value in gain_initializers:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be an int or float")
            try:
                value = float(value)
            except OverflowError as error:
                raise ValueError(f"{name} must be finite") from error
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            validated_gains[name] = value

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_tokens = num_tokens
        self.head_dim = d_model // num_heads
        self.activation_name = activation
        self.group_chunk_size = group_chunk_size

        self.query_proj = nn.Linear(d_model, d_model, bias=True)
        self.key_proj = nn.Linear(d_model, d_model, bias=True)
        self.value_proj = nn.Linear(d_model, d_model, bias=True)
        self.output_proj = nn.Linear(d_model, d_model, bias=False)

        self.attention_weight_dropout = nn.Dropout(dropout)
        self.attention_output_dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn_linear1 = nn.Linear(d_model, dim_feedforward)
        self.ffn_linear2 = nn.Linear(dim_feedforward, d_model)
        self.ffn_output_dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        self.relation_gain = nn.Parameter(
            torch.tensor(validated_gains["relation_gain_init"])
        )
        self.block_gain = nn.Parameter(
            torch.tensor(validated_gains["block_gain_init"])
        )

    def _activation(self, hidden_states):
        if self.activation_name == "relu":
            return F.relu(hidden_states)
        if self.activation_name == "gelu":
            return F.gelu(hidden_states)
        return F.silu(hidden_states)

    def _forward_chunk(self, region_scale_states, corr):
        groups = region_scale_states.shape[0]

        query = self.query_proj(region_scale_states)
        key = self.key_proj(region_scale_states)
        value = self.value_proj(region_scale_states)

        query = query.reshape(
            groups,
            self.num_tokens,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)
        key = key.reshape(
            groups,
            self.num_tokens,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)
        value = value.reshape(
            groups,
            self.num_tokens,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        content_logits = torch.matmul(query, key.transpose(-2, -1))
        content_logits = content_logits / math.sqrt(self.head_dim)
        relation_logits = content_logits + self.relation_gain * corr
        attention_prob = torch.softmax(relation_logits, dim=-1)
        attention_prob = self.attention_weight_dropout(attention_prob)

        attention_output = torch.matmul(attention_prob, value)
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.reshape(
            groups,
            self.num_tokens,
            self.d_model,
        )
        attention_output = self.output_proj(attention_output)

        z_attn = self.norm1(
            region_scale_states
            + self.attention_output_dropout(attention_output)
        )
        ffn_output = self.ffn_linear2(
            self._activation(self.ffn_linear1(z_attn))
        )
        z_block = self.norm2(
            z_attn + self.ffn_output_dropout(ffn_output)
        )

        return torch.lerp(region_scale_states, z_block, self.block_gain)

    def forward(self, region_scale_states, corr):
        if not isinstance(region_scale_states, torch.Tensor):
            raise TypeError("region_scale_states must be a Tensor")
        if not isinstance(corr, torch.Tensor):
            raise TypeError("corr must be a Tensor")
        if region_scale_states.ndim != 3:
            raise ValueError("region_scale_states must be a rank-3 Tensor [G,12,D]")
        if corr.ndim != 4:
            raise ValueError("corr must be a rank-4 Tensor [G,M,12,12]")

        groups, num_tokens, hidden_dim = region_scale_states.shape
        corr_groups, corr_heads, corr_rows, corr_columns = corr.shape
        if groups != corr_groups:
            raise ValueError("region_scale_states and corr must have the same G")
        if num_tokens != self.num_tokens:
            raise ValueError("region_scale_states must contain exactly 12 tokens")
        if corr_rows != self.num_tokens or corr_columns != self.num_tokens:
            raise ValueError("corr relation axes must be 12 x 12")
        if corr_heads != self.num_heads:
            raise ValueError("corr head dimension must equal num_heads")
        if hidden_dim != self.d_model:
            raise ValueError("region_scale_states last dimension must equal d_model")
        if region_scale_states.device != corr.device:
            raise ValueError("region_scale_states and corr must share a device")
        if region_scale_states.dtype != corr.dtype:
            raise TypeError("region_scale_states and corr must share a dtype")
        if groups == 0:
            raise ValueError("G must be positive")

        if self.group_chunk_size is None or self.group_chunk_size >= groups:
            return self._forward_chunk(region_scale_states, corr)

        output_chunks = []
        for start in range(0, groups, self.group_chunk_size):
            end = min(start + self.group_chunk_size, groups)
            output_chunks.append(
                self._forward_chunk(
                    region_scale_states[start:end],
                    corr[start:end],
                )
            )
        return torch.cat(output_chunks, dim=0)
