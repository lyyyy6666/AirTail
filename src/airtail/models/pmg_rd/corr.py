import torch
import torch.nn as nn


class CrossKnowledgeCorrelation(nn.Module):
    """逐头计算两组 Guider 原始分数的双线性相关矩阵。

    输入：
        temporal_scores: ``[G,M,12,12]``。
        variable_scores: ``[G,M,12,12]``。

    输出：
        corr: ``[G,M,12,12]``。
    """

    def __init__(self, num_heads, num_tokens=12):
        super().__init__()
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if num_tokens != 12:
            raise ValueError("num_tokens must be 12 for Relation-KD V2")

        self.num_heads = num_heads
        self.num_tokens = num_tokens
        self.mode = "cross_evidence"
        self.correlation_weight = nn.Parameter(torch.eye(num_tokens))

    def forward(self, temporal_scores, variable_scores):
        if not isinstance(temporal_scores, torch.Tensor):
            raise TypeError("temporal_scores must be a Tensor")
        if not isinstance(variable_scores, torch.Tensor):
            raise TypeError("variable_scores must be a Tensor")
        if temporal_scores.ndim != 4:
            raise ValueError("temporal_scores must be a rank-4 Tensor [G,M,12,12]")
        if variable_scores.ndim != 4:
            raise ValueError("variable_scores must be a rank-4 Tensor [G,M,12,12]")
        if temporal_scores.shape != variable_scores.shape:
            raise ValueError("temporal_scores and variable_scores must have the same shape")

        _, num_heads, query_tokens, knowledge_tokens = temporal_scores.shape
        if num_heads != self.num_heads:
            raise ValueError("input head dimension must equal num_heads")
        if query_tokens != self.num_tokens or knowledge_tokens != self.num_tokens:
            raise ValueError("both input relation axes must be 12 x 12")
        if temporal_scores.dtype != variable_scores.dtype:
            raise TypeError("temporal_scores and variable_scores must share a dtype")
        if temporal_scores.device != variable_scores.device:
            raise ValueError("temporal_scores and variable_scores must share a device")

        def bilinear(left, right):
            return torch.einsum(
                "gmik,kl,gmjl->gmij",
                left,
                self.correlation_weight,
                right,
            )

        # cross_evidence: variable_scores participates transposed in the
        # per-head bilinear product used by the frozen Full implementation.
        return bilinear(temporal_scores, variable_scores)
