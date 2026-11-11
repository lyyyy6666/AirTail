import math

import torch
import torch.nn as nn


class RegionScaleRelationGuider(nn.Module):
    """计算区域–尺度状态到知识 Token 的多头原始关联分数。

    输入：
        region_scale_states: ``[G,12,D]``。
        knowledge_tokens: ``[G,12,D]``。

    输出：
        raw_scores: ``[G,M,12,12]``，未经过 softmax。
    """

    def __init__(self, d_model, num_heads, num_tokens=12):
        super().__init__()
        if d_model <= 0 or num_heads <= 0:
            raise ValueError("d_model and num_heads must be positive")
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if num_tokens != 12:
            raise ValueError("num_tokens must be 12 for Relation-KD V2")

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_tokens = num_tokens
        self.head_dim = d_model // num_heads

        self.query_proj = nn.Linear(d_model, d_model, bias=False)
        self.key_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, region_scale_states, knowledge_tokens):
        if not isinstance(region_scale_states, torch.Tensor):
            raise TypeError("region_scale_states must be a Tensor")
        if not isinstance(knowledge_tokens, torch.Tensor):
            raise TypeError("knowledge_tokens must be a Tensor")
        if region_scale_states.ndim != 3:
            raise ValueError("region_scale_states must be a rank-3 Tensor [G,12,D]")
        if knowledge_tokens.ndim != 3:
            raise ValueError("knowledge_tokens must be a rank-3 Tensor [G,12,D]")

        state_groups, state_tokens, state_dim = region_scale_states.shape
        knowledge_groups, knowledge_count, knowledge_dim = knowledge_tokens.shape
        if state_groups != knowledge_groups:
            raise ValueError("region_scale_states and knowledge_tokens must have the same G")
        if state_tokens != self.num_tokens or knowledge_count != self.num_tokens:
            raise ValueError("both inputs must contain exactly 12 tokens")
        if state_dim != self.d_model or knowledge_dim != self.d_model:
            raise ValueError("both input last dimensions must equal d_model")
        if region_scale_states.device != knowledge_tokens.device:
            raise ValueError("region_scale_states and knowledge_tokens must share a device")
        if region_scale_states.dtype != knowledge_tokens.dtype:
            raise TypeError("region_scale_states and knowledge_tokens must share a dtype")

        # 分别投影状态 Query 与知识 Key，再保留独立的多头维度。
        query = self.query_proj(region_scale_states)
        key = self.key_proj(knowledge_tokens)
        query = query.reshape(
            state_groups,
            self.num_tokens,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)
        key = key.reshape(
            state_groups,
            self.num_tokens,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        # 每个 G group、每个 head 独立计算未归一化关联分数。
        return torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(
            self.head_dim
        )
