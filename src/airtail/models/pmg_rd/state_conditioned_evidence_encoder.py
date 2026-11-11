import math

import torch
import torch.nn as nn


class StateConditionedEvidenceEncoder(nn.Module):
    """Construct evidence tokens by querying evidence with region-scale states.

    Inputs:
        state_tokens: ``[G,12,D]`` Teacher region-scale states.
        evidence_tokens: ``[G,F,D]`` temporal or variable evidence.
        evidence_mask: ``[G,F]`` bool mask, True for valid evidence.

    Output:
        encoded_tokens: ``[G,12,D]`` state-conditioned evidence tokens.
    """

    def __init__(
        self,
        d_model,
        num_heads,
        dropout,
        num_tokens=12,
        query_source="state",
    ):
        super().__init__()
        if d_model <= 0 or num_heads <= 0:
            raise ValueError("d_model and num_heads must be positive")
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if not 0.0 <= dropout <= 1.0:
            raise ValueError("dropout must be between 0 and 1")
        if num_tokens != 12:
            raise ValueError("num_tokens must be 12 for Relation-KD V2")
        if query_source not in {"state", "learned"}:
            raise ValueError("query_source must be state or learned")

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_tokens = num_tokens
        self.head_dim = d_model // num_heads
        self.query_source = query_source
        self.learned_queries = (
            nn.Parameter(torch.empty(num_tokens, d_model))
            if query_source == "learned"
            else None
        )

        self.query_norm = nn.LayerNorm(d_model)
        self.query_projection = nn.Linear(d_model, d_model, bias=False)
        self.key_projection = nn.Linear(d_model, d_model, bias=False)
        self.value_projection = nn.Linear(d_model, d_model, bias=False)
        self.output_projection = nn.Linear(d_model, d_model, bias=False)
        self.attention_weight_dropout = nn.Dropout(dropout)
        if self.learned_queries is not None:
            nn.init.normal_(self.learned_queries, mean=0.0, std=0.02)

    def forward(self, state_tokens, evidence_tokens, evidence_mask):
        if not isinstance(state_tokens, torch.Tensor):
            raise TypeError("state_tokens must be a Tensor")
        if not isinstance(evidence_tokens, torch.Tensor):
            raise TypeError("evidence_tokens must be a Tensor")
        if not isinstance(evidence_mask, torch.Tensor):
            raise TypeError("evidence_mask must be a Tensor")
        if state_tokens.ndim != 3:
            raise ValueError("state_tokens must have shape [G,12,D]")
        if evidence_tokens.ndim != 3:
            raise ValueError("evidence_tokens must have shape [G,F,D]")

        groups, num_tokens, state_dim = state_tokens.shape
        evidence_groups, evidence_length, evidence_dim = evidence_tokens.shape
        if groups == 0:
            raise ValueError("G must be positive")
        if num_tokens != self.num_tokens or state_dim != self.d_model:
            raise ValueError("state_tokens must have shape [G,12,D]")
        if evidence_groups != groups or evidence_dim != self.d_model:
            raise ValueError("evidence_tokens must have shape [G,F,D]")
        if evidence_length == 0:
            raise ValueError("evidence length F must be positive")
        if evidence_mask.shape != (groups, evidence_length):
            raise ValueError("evidence_mask must have shape [G,F]")
        if evidence_mask.dtype != torch.bool:
            raise TypeError("evidence_mask must have bool dtype")
        if not state_tokens.is_floating_point() or not evidence_tokens.is_floating_point():
            raise TypeError("state_tokens and evidence_tokens must use floating point dtypes")
        if state_tokens.dtype != evidence_tokens.dtype:
            raise TypeError("state_tokens and evidence_tokens must share a dtype")
        if state_tokens.device != evidence_tokens.device:
            raise ValueError("state_tokens and evidence_tokens must share a device")
        if evidence_mask.device != state_tokens.device:
            raise ValueError("evidence_mask and inputs must share a device")

        parameter = self.query_projection.weight
        if parameter.dtype != state_tokens.dtype:
            raise TypeError("inputs dtype must match module dtype")
        if parameter.device != state_tokens.device:
            raise ValueError("inputs device must match module device")

        safe_evidence = torch.where(
            evidence_mask.unsqueeze(-1),
            evidence_tokens,
            torch.zeros_like(evidence_tokens),
        )
        active_groups = evidence_mask.any(dim=1)

        # Allow one zero-valued token in all-invalid groups so softmax stays finite.
        safe_mask = evidence_mask.clone()
        safe_mask[~active_groups, 0] = True

        query_tokens = (
            state_tokens
            if self.query_source == "state"
            else self.learned_queries[None].expand(groups, -1, -1)
        )
        query = self.query_projection(self.query_norm(query_tokens))
        key = self.key_projection(safe_evidence)
        value = self.value_projection(safe_evidence)

        query = query.reshape(
            groups, self.num_tokens, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key = key.reshape(
            groups, evidence_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        value = value.reshape(
            groups, evidence_length, self.num_heads, self.head_dim
        ).transpose(1, 2)

        logits = torch.matmul(query, key.transpose(-2, -1))
        logits = logits / math.sqrt(self.head_dim)
        logits = logits.masked_fill(~safe_mask[:, None, None, :], float("-inf"))
        attention = torch.softmax(logits, dim=-1)
        attention = self.attention_weight_dropout(attention)

        encoded = torch.matmul(attention, value)
        encoded = encoded.transpose(1, 2).contiguous().reshape(
            groups, self.num_tokens, self.d_model
        )
        encoded = self.output_projection(encoded)

        # The temporary safe token is numerical only; invalid groups stay exact zero.
        return encoded * active_groups[:, None, None].to(encoded.dtype)
