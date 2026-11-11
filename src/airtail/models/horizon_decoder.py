"""
QC-MSSM 的固定容量预测时域解码器（完整中文注释版）。

本文件只增加解释性注释，不改变原始计算逻辑。

统一使用以下维度记号：

    B：batch size，批量大小
    L：历史序列长度
    N：站点数量
    R：分位区域数量，本模型固定为 3
       （median / upper / extreme）
    S：多尺度分支数量
    D：隐藏特征维度，即 d_model
    H：未来预测长度，即 horizon

上一个 QuantileConditionedMultiscaleSSM 模块的输出为：

    states: [B, L, N, R, S, D]

本解码器的主要任务是：

    1. 沿历史时间维 L 汇总每条历史状态；
    2. 为未来 H 个预测位置加入不同的预测时域编码；
    3. 保留站点维 N 和分位区域维 R；
    4. 沿多尺度维 S 做融合。

最终主要输出为：

    fused: [B, H, N, R, D]

注意：
这里不是自回归地逐步生成未来，也不会把前一个预测值作为下一个预测
时刻的输入。它先把历史压缩成 summary，再利用 horizon encoding 一次性
构造所有 H 个未来位置的隐藏表示。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_horizon_encoding(horizon, channels, device, dtype):
    """
    为未来第 0, 1, ..., H-1 个预测位置生成固定的正弦/余弦编码。

    参数
    ----
    horizon:
        未来预测长度 H。
    channels:
        隐藏特征维度 D。
    device:
        编码需要放置的设备，例如 CPU 或 CUDA。
    dtype:
        最终编码需要使用的数据类型，例如 float32 或 float16。

    返回
    ----
    encoding:
        形状为 [H,D] 的固定预测位置编码。

    作用
    ----
    历史 summary 本身对每个未来位置都是相同的。如果不加入 horizon
    encoding，解码器就无法区分“预测第 1 步”和“预测第 24 步”。

    这个编码没有可学习参数，但后面会经过可学习的 horizon_proj。
    """

    # 未来位置编号：
    #
    # [0, 1, 2, ..., H-1] -> [H,1]
    #
    # 额外增加的最后一维便于与不同频率广播相乘。
    # 这里先用 float32 计算三角函数，以保持数值稳定性。
    position = torch.arange(
        horizon, device=device, dtype=torch.float32
    )[:, None]

    # 取隐藏通道中的偶数位置：
    #
    # D=6 时：even = [0,2,4]
    # D=5 时：even = [0,2,4]
    #
    # 数量等于 ceil(D/2)，分别对应 encoding 的 0、2、4...列。
    even = torch.arange(
        0, channels, 2, device=device, dtype=torch.float32
    )

    # 为不同通道生成从高频到低频的频率系数。
    #
    # 较小的通道索引变化更快，较大的通道索引变化更慢，
    # 从而让不同预测位置拥有可区分的多频率表示。
    frequency = torch.exp(
        -math.log(10000.0) * even / max(channels, 1)
    )

    # 先创建空编码矩阵：
    #
    # [H,D]
    encoding = torch.zeros(
        horizon, channels, device=device, dtype=torch.float32
    )

    # 偶数通道写入 sin：
    #
    # position:  [H,1]
    # frequency: [ceil(D/2)]
    # 广播相乘后为 [H,ceil(D/2)]。
    encoding[:, 0::2] = torch.sin(position * frequency)

    # 奇数通道写入 cos。
    #
    # 当 D 为奇数时，奇数列比偶数列少一个，因此需要截取 frequency，
    # 使右侧列数与 encoding[:, 1::2] 完全一致。
    if channels > 1:
        encoding[:, 1::2] = torch.cos(
            position
            * frequency[: encoding[:, 1::2].shape[1]]
        )

    # 三角函数在 float32 中计算完后，再转换成 states 使用的 dtype。
    return encoding.to(dtype=dtype)


class FixedCapacityHorizonDecoder(nn.Module):
    """
    将历史多区域、多尺度状态解码为固定预测时域表示。

    输入：
        states [B,L,N,R,S,D]

    输出：
        fused               [B,H,N,R,D]
        scale_weights       [B,H,N,R,S]
        horizon_scale_states[B,H,N,R,S,D]

    “FixedCapacity”的直观含义是：
    解码器使用共享的 horizon_proj 和固定正弦编码处理所有未来位置，
    而不是为 H 个位置分别创建一整套独立网络。因此，核心网络容量不会
    随 horizon 线性扩张。
    """

    def __init__(
        self,
        d_model,
        num_scales,
        horizon,
        scale_fusion="horizon_region",
    ):
        """
        初始化预测时域解码器。

        参数
        ----
        d_model:
            隐藏特征维度 D。
        num_scales:
            多尺度分支数量 S。
        horizon:
            未来预测长度 H。
        scale_fusion:
            多尺度融合方式：

            - horizon_region：
              为每个样本、未来位置、站点和分位区域，动态学习 S 个
              尺度权重。

            - uniform：
              所有尺度始终使用相同权重 1/S。

            - single_scale：
              只使用第 0 个尺度，其余尺度权重固定为 0。

            后两种模式主要可以作为多尺度动态融合的对照/消融。
        """
        super().__init__()

        # 只允许三种已经实现的尺度融合模式。
        if scale_fusion not in {
            "horizon_region",
            "uniform",
            "single_scale",
        }:
            raise ValueError(
                f"unsupported scale fusion: {scale_fusion}"
            )

        # 保存模型维度和配置。
        self.d_model = int(d_model)
        self.num_scales = int(num_scales)

        # 前面的 QC-MSSM 固定使用三个分位区域：
        # median / upper / extreme。
        self.num_regions = 3

        self.horizon = int(horizon)
        self.scale_fusion = scale_fusion

        # attention_score：
        # 为历史序列中每个时间步的状态生成一个标量注意力分数。
        #
        # [B,L,N,R,S,D] -> [B,L,N,R,S,1]
        #
        # 后面会在历史时间维 L 上做 softmax，再得到加权历史摘要。
        self.attention_score = nn.Linear(d_model, 1)

        # summary_proj：
        # 把三种互补的历史摘要拼接后压回 D 维。
        #
        # 三种摘要分别是：
        # 1. 最后一个有效历史状态 last；
        # 2. 所有有效历史状态的均值 mean；
        # 3. 注意力加权历史状态 attended。
        #
        # [last,mean,attended]：
        # [B,N,R,S,3D] -> [B,N,R,S,D]
        self.summary_proj = nn.Linear(3 * d_model, d_model)

        # 三个分位区域各自拥有一个可学习嵌入：
        #
        # [R,D] = [3,D]
        #
        # 它让解码阶段仍能区分 median、upper 和 extreme。
        self.region_embedding = nn.Parameter(
            torch.randn(self.num_regions, d_model) * 0.02
        )

        # 每个尺度各自拥有一个可学习嵌入：
        #
        # [S,D]
        #
        # 即使各尺度的历史 summary 数值接近，这个嵌入也会告诉解码器
        # 当前状态来自哪一个尺度。
        self.scale_embedding = nn.Parameter(
            torch.randn(num_scales, d_model) * 0.02
        )

        # horizon_proj：
        # 将固定的正弦/余弦预测位置编码转换为可学习的未来位置表示。
        #
        # [H,D] -> [H,D]
        #
        # 这里同一个小网络共享给所有 H 个未来位置。
        self.horizon_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # scale_score：
        # 在 horizon_region 模式下，为每个尺度状态生成一个标量分数。
        #
        # [B,H,N,R,S,D] -> [B,H,N,R,S,1]
        #
        # 后面在尺度维 S 上做 softmax，得到尺度融合权重。
        self.scale_score = nn.Linear(d_model, 1)

        # 对每个未来状态的最后一个隐藏维 D 做归一化。
        self.norm = nn.LayerNorm(d_model)

    def _station_mask(self, history_mask, states):
        """
        把输入的历史有效性掩码统一为站点级布尔掩码 [B,L,N]。

        history_mask 可以是：

            [B,L,N,1]：每个站点、时间步一个有效标记；
            [B,L,N,F]：每个原始特征一个有效标记。

        如果最后一维中至少有一个元素有效，该站点在当前时间步就被
        视为拥有有效状态。
        """

        # 没有显式提供 mask 时，默认所有历史时间步都有效。
        #
        # states.shape[:3] = [B,L,N]
        if history_mask is None:
            return torch.ones(
                *states.shape[:3],
                device=states.device,
                dtype=torch.bool,
            )

        # history_mask 必须有四个维度，并且 B、L、N 与 states 对齐。
        if (
            history_mask.ndim != 4
            or history_mask.shape[:3] != states.shape[:3]
        ):
            raise ValueError(
                "history_mask must have shape "
                "[B,L,N,1] or [B,L,N,F]"
            )

        # 先转成 bool，再沿最后一个特征维做 any：
        #
        # [B,L,N,1/F] -> [B,L,N]
        return history_mask.to(dtype=torch.bool).any(dim=-1)

    def _temporal_summary(self, states, history_mask):
        """
        沿历史时间维 L 汇总状态。

        输入：
            states [B,L,N,R,S,D]

        返回：
            summary [B,N,R,S,D]

        该函数并不是只取最后一个时间步，而是同时使用：

            last     最后一个有效状态；
            mean     有效状态平均值；
            attended 学习得到的注意力加权状态。
        """

        # 读取各维度大小，便于构造后续索引。
        (
            batch,
            length,
            stations,
            regions,
            scales,
            channels,
        ) = states.shape

        # 统一成站点级布尔掩码：
        #
        # [B,L,N]
        mask = self._station_mask(history_mask, states)

        # 统计每个“样本×站点”拥有多少个有效历史时间步：
        #
        # [B,L,N] -> [B,N]
        valid_count = mask.sum(dim=1)

        # 如果某个站点整段历史都无效，就无法构造 last、mean 和
        # attention summary，因此直接报错，避免静默产生 NaN。
        if (valid_count == 0).any():
            raise ValueError(
                "each station history must contain "
                "at least one valid state"
            )

        # 构造历史时间索引：
        #
        # [0,1,...,L-1] -> [1,L,1]
        #
        # 后续可广播到 [B,L,N]。
        time_index = torch.arange(
            length, device=states.device
        )[None, :, None]

        # 对有效位置保留其时间索引，无效位置改成 -1，
        # 再沿 L 取最大值，得到每个站点最后一个有效时间步的位置：
        #
        # [B,L,N] -> [B,N]
        last_index = torch.where(
            mask, time_index, -1
        ).amax(dim=1)

        # states.gather 需要索引张量与输出形状对齐。
        #
        # [B,N]
        #   -> 增加 L、R、S、D 对应的维度
        # [B,1,N,1,1,1]
        #   -> 扩展
        # [B,1,N,R,S,D]
        gather_index = last_index[
            :, None, :, None, None, None
        ].expand(
            -1,
            1,
            -1,
            regions,
            scales,
            channels,
        )

        # 沿历史时间维 L 取出最后一个有效状态：
        #
        # gather 后：[B,1,N,R,S,D]
        # squeeze 后：[B,N,R,S,D]
        last = states.gather(
            dim=1, index=gather_index
        ).squeeze(1)

        # 把 mask 从 [B,L,N] 扩展为可与 states 广播的：
        #
        # [B,L,N,1,1,1]
        expanded_mask = mask[..., None, None, None]

        # 只对有效历史状态求平均：
        #
        # states * expanded_mask：
        # [B,L,N,R,S,D]
        #
        # 沿 L 求和后：
        # [B,N,R,S,D]
        #
        # valid_count[...,None,None,None]：
        # [B,N,1,1,1]
        #
        # 最终 mean：
        # [B,N,R,S,D]
        mean = (
            states * expanded_mask
        ).sum(dim=1) / valid_count[..., None, None, None]

        # 为每个历史时间步生成注意力分数：
        #
        # [B,L,N,R,S,D]
        #   -> Linear(D,1)
        # [B,L,N,R,S,1]
        #   -> squeeze
        # [B,L,N,R,S]
        scores = self.attention_score(
            states
        ).squeeze(-1)

        # 无效时间步的分数设为 -inf。
        #
        # mask[...,None,None]：
        # [B,L,N,1,1]
        #
        # 它会广播到三个区域 R 和所有尺度 S。
        # 这样 softmax 后，无效位置的注意力权重严格为 0。
        scores = scores.masked_fill(
            ~mask[..., None, None], -torch.inf
        )

        # 在历史时间维 L 上做 softmax：
        #
        # weights [B,L,N,R,S]
        #
        # 对固定的 B、N、R、S 而言，其 L 个权重之和为 1。
        weights = F.softmax(scores, dim=1)

        # 用注意力权重对所有历史状态加权求和：
        #
        # weights[...,None]：[B,L,N,R,S,1]
        # states：           [B,L,N,R,S,D]
        #
        # 沿 L 求和后：
        # attended [B,N,R,S,D]
        attended = (
            weights[..., None] * states
        ).sum(dim=1)

        # 拼接三种历史摘要：
        #
        # last / mean / attended 均为 [B,N,R,S,D]
        #
        # 拼接后：[B,N,R,S,3D]
        # 投影后：[B,N,R,S,D]
        return self.summary_proj(
            torch.cat(
                (last, mean, attended), dim=-1
            )
        )

    def build_horizon_scale_states(
        self, states, history_mask=None
    ):
        """
        汇总历史并构造 LayerNorm 后的未来多尺度状态。

        参数
        ----
        states:
            QC-MSSM 输出的历史状态 [B,L,N,R,S,D]。
        history_mask:
            可选历史有效性掩码 [B,L,N,1] 或 [B,L,N,F]。

        返回 LayerNorm 后的 horizon_scale_states [B,H,N,R,S,D]。
        """

        # 输入必须严格包含：
        #
        # B、L、N、R、S、D 六个维度。
        if states.ndim != 6:
            raise ValueError(
                "horizon decoder expects [B,L,N,R,S,D]"
            )

        # 沿历史时间维 L 汇总：
        #
        # [B,L,N,R,S,D] -> [B,N,R,S,D]
        #
        # 此时历史长度 L 被压缩掉，但站点、区域和尺度仍全部保留。
        summary = self._temporal_summary(
            states, history_mask
        )

        # 为未来 H 个位置生成固定正弦/余弦编码：
        #
        # [H,D]
        horizon_encoding = sinusoidal_horizon_encoding(
            self.horizon,
            self.d_model,
            states.device,
            states.dtype,
        )

        # 用共享的可学习 MLP 转换未来位置编码：
        #
        # [H,D] -> [H,D]
        horizon_term = self.horizon_proj(
            horizon_encoding
        )

        # 把四类信息相加，构造未来每个位置、站点、区域、尺度的状态：
        #
        # 1. summary[:,None]
        #    [B,1,N,R,S,D]
        #    同一份历史摘要广播到 H 个未来位置。
        #
        # 2. horizon_term[None,:,None,None,None]
        #    [1,H,1,1,1,D]
        #    告诉模型当前是第几个未来预测位置。
        #
        # 3. region_embedding[None,None,None,:,None]
        #    [1,1,1,R,1,D]
        #    告诉模型当前属于 median / upper / extreme 哪个区域。
        #
        # 4. scale_embedding[None,None,None,None]
        #    [1,1,1,1,S,D]
        #    告诉模型当前来自哪个时间尺度。
        #
        # 广播相加后：
        # [B,H,N,R,S,D]
        horizon_scale_states = (
            summary[:, None]
            + horizon_term[
                None, :, None, None, None
            ]
            + self.region_embedding[
                None, None, None, :, None
            ]
            + self.scale_embedding[
                None, None, None, None
            ]
        )

        # 对最后一个隐藏特征维 D 做 LayerNorm。
        #
        # 形状保持为 [B,H,N,R,S,D]。
        horizon_scale_states = self.norm(
            horizon_scale_states
        )

        return horizon_scale_states

    def aggregate_scales(
        self, horizon_scale_states, reference_states
    ):
        """
        按现有 fusion 分支聚合尺度并保持原返回结构。
        """

        if self.scale_fusion == "horizon_region":
            # 动态多尺度融合。
            #
            # tanh 不改变形状，只把隐藏值压缩到 (-1,1)，
            # 再由 scale_score 为每个尺度产生一个标量分数：
            #
            # [B,H,N,R,S,D]
            #   -> [B,H,N,R,S,1]
            #   -> [B,H,N,R,S]
            logits = self.scale_score(
                torch.tanh(horizon_scale_states)
            ).squeeze(-1)

            # 在尺度维 S 上做 softmax。
            #
            # 对固定的 B、H、N、R 而言，所有尺度权重之和为 1。
            #
            # 因此，不同未来位置和不同分位区域可以选择不同尺度：
            #
            # 例如，较近预测位置可能更依赖短尺度，
            # extreme 区域可能学习到更依赖长尺度。
            scale_weights = F.softmax(
                logits, dim=-1
            )

        elif self.scale_fusion == "uniform":
            # 均匀融合：
            #
            # scale_weights [B,H,N,R,S]
            #
            # 所有尺度权重都固定为 1/S，不依赖数据、预测位置或区域。
            scale_weights = reference_states.new_full(
                horizon_scale_states.shape[:-1],
                1.0 / self.num_scales,
            )

        else:
            # single_scale 模式：
            #
            # 先把所有尺度权重设成 0，再令第 0 个尺度权重为 1。
            # 因此只保留第一个尺度。
            scale_weights = reference_states.new_zeros(
                horizon_scale_states.shape[:-1]
            )
            scale_weights[..., 0] = 1.0

        # 用尺度权重做加权求和：
        #
        # scale_weights[...,None]：[B,H,N,R,S,1]
        # horizon_scale_states：   [B,H,N,R,S,D]
        #
        # 相乘后沿 dim=4（尺度维 S）求和：
        #
        # fused [B,H,N,R,D]
        #
        # 至此：
        # - 历史时间维 L 已被汇总；
        # - 未来时间维 H 已被构造；
        # - 多尺度维 S 已被融合；
        # - 站点维 N 和分位区域维 R 仍然保留。
        fused = (
            scale_weights[..., None]
            * horizon_scale_states
        ).sum(dim=4)

        # 同时返回：
        #
        # 1. 融合后的主输出；
        # 2. 尺度融合权重；
        # 3. 融合前的完整多尺度未来状态。
        return (
            fused,
            scale_weights,
            horizon_scale_states,
        )

    def forward(self, states, history_mask=None):
        """
        执行历史汇总、未来位置展开和多尺度融合。

        参数
        ----
        states:
            QC-MSSM 输出的历史状态 [B,L,N,R,S,D]。
        history_mask:
            可选历史有效性掩码 [B,L,N,1] 或 [B,L,N,F]。

        返回
        ----
        fused:
            融合尺度后的未来隐藏状态 [B,H,N,R,D]。
            这是后续 quantile head 的主要输入。

        scale_weights:
            每个尺度的融合权重 [B,H,N,R,S]。
            可用于诊断不同预测时域和分位区域偏好的尺度。

        horizon_scale_states:
            融合前的完整未来多尺度状态 [B,H,N,R,S,D]。
            保留它便于诊断或其他辅助损失使用。
        """
        horizon_scale_states = self.build_horizon_scale_states(
            states, history_mask
        )
        return self.aggregate_scales(
            horizon_scale_states, states
        )
