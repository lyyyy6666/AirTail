
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalMultiscaleTemporalStem(nn.Module):
    """多尺度因果时间特征提取模块。

    输入形状：
        [B, L, N, D]

    输出形状：
        [B, L, N, S, D]

    其中：
        B：批次大小
        L：历史序列长度
        N：站点数量
        D：隐藏特征维度
        S：时间尺度数量，即卷积核数量

    每个时间尺度使用一个独立的一维卷积。卷积只沿时间维进行，
    不在该模块中进行站点之间的信息交互。
    """

    def __init__(self, d_model, kernels):
        super().__init__()

        # 每个时间尺度必须对应一个正整数卷积核
        if not kernels or min(kernels) <= 0:
            raise ValueError("kernels must contain positive integers")

        # 保存各个时间尺度的卷积核大小，例如 (1, 3, 7, 24)
        self.kernels = tuple(int(kernel) for kernel in kernels)

        # 每个时间尺度使用一组独立的一维卷积
        # 输入与输出通道数均为 d_model
        # groups 默认为 1，因此卷积会同时进行时间聚合和隐藏通道混合
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=d_model,
                    out_channels=d_model,
                    kernel_size=kernel,
                    padding=0,
                )
                for kernel in self.kernels
            ]
        )

        # 每个时间尺度分别使用独立的 LayerNorm
        self.norms = nn.ModuleList(
            [nn.LayerNorm(d_model) for _ in self.kernels]
        )

    @property
    def num_scales(self):
        """返回时间尺度数量。"""
        return len(self.kernels)

    def forward(self, x):
        # 输入必须采用 [批次, 时间, 站点, 隐藏维度] 的布局
        if x.ndim != 4:
            raise ValueError(
                f"temporal stem expects [B,L,N,D], got {tuple(x.shape)}"
            )

        batch, length, stations, channels = x.shape

        # 将每个站点视为一个独立的时间序列：
        # [B, L, N, D] → [B, N, D, L] → [B*N, D, L]
        #
        # Conv1d 要求输入布局为：
        # [样本数, 通道数, 序列长度]
        temporal = (
            x.permute(0, 2, 3, 1)
            .reshape(batch * stations, channels, length)
        )

        scales = []

        # 分别提取不同时间感受野下的因果特征
        for kernel, conv, norm in zip(
            self.kernels,
            self.convs,
            self.norms,
        ):
            # 只在时间序列左侧补充 kernel - 1 个零
            # 保证时刻 t 的输出只依赖当前及过去时刻，不使用未来信息
            padded = F.pad(temporal, (kernel - 1, 0))

            # 沿时间维执行一维卷积
            # [B*N, D, L+kernel-1] → [B*N, D, L]
            y = conv(padded)

            # 恢复原始批次和站点维度：
            # [B*N, D, L] → [B, N, D, L] → [B, L, N, D]
            y = (
                y.reshape(batch, stations, channels, length)
                .permute(0, 3, 1, 2)
            )

            # 依次执行 SiLU 激活和隐藏维度归一化
            y = norm(F.silu(y))

            scales.append(y)

        # 在第 4 个维度堆叠不同时间尺度：
        # S 个 [B, L, N, D] → [B, L, N, S, D]
        # 此处只保留多尺度结果，不进行尺度融合
        return torch.stack(scales, dim=3)
