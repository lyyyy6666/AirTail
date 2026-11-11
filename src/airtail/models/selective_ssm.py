"""
带中文讲解的 Quantile-Conditioned Multiscale SSM。

这份文件只增加解释性注释，没有改变原始代码的计算逻辑。

建议先记住输入和输出的核心形状：

    输入 x:
        [B, L, N, S, D]

    scan 前加入分位区域 R 后:
        [B, L, N, R, S, D]

    输出 y:
        [B, L, N, R, S, D]

其中：
    B = batch size，批次大小
    L = history length，历史序列长度
    N = number of stations，站点数
    R = number of quantile regions，分位区域数，这里固定为 3
    S = number of scales，多时间尺度数量
    D = d_model，隐藏特征维度

本模块的主要作用不是直接输出最终 q50、q75、q90、q95，而是：

1. 为 median、upper、extreme 三个分位区域构造不同的记忆速度 delta；
2. 在每个站点、尺度和分位区域上分别执行 selective scan；
3. 输出带有“分位区域 × 多尺度”结构的时序表示，供后续预测头使用。
"""

import math

import torch
import torch.nn as nn

from airtail.models.scan import bounded_selective_scan


# 三个区域不是三个具体分位数，而是三类分位语义：
# - median：中位/主体区域
# - upper：上尾区域
# - extreme：更极端的上尾区域
#
# 后续代码只使用它们的顺序和数量。名称主要用于说明模型设计。
REGION_NAMES = ("median", "upper", "extreme")


class QuantileConditionedMultiscaleSSM(nn.Module):
    """带分位区域条件和多尺度条件的状态空间时序模块。"""

    def __init__(
        self,
        d_model,
        d_state,
        num_scales,
        delta_min,
        delta_max,
        delta_mode="scale_region_history",
        history_gamma=1.0,
        backend="reference",
    ):
        """
        初始化模型结构与可学习参数。

        参数
        ----
        d_model:
            输入和输出的隐藏特征维度 D，同时也是 selective scan 的通道数。
        d_state:
            每个隐藏通道内部的状态槽数量，即 SSM 的 state 维度。
        num_scales:
            多尺度分支数量 S。
        delta_min, delta_max:
            delta 的严格上下界。最终所有 delta 都会落在这个区间内。
        delta_mode:
            控制生成 delta 时使用哪些条件信息：

            - vanilla:
              只根据当前输入 x 生成 delta。
            - scale_only:
              当前输入 + 尺度偏置。
            - scale_region:
              当前输入 + 尺度偏置 + 分位区域/尺度联合偏置。
            - scale_region_history:
              在 scale_region 基础上，再加入历史汇总信息。
        history_gamma:
            历史修正项的强度系数。
        backend:
            selective scan 的执行后端：
            - reference：普通 PyTorch 逐时间步实现，便于检查；
            - cuda：调用编译好的 CUDA selective-scan 扩展。
        """
        super().__init__()

        # delta 必须为有限正数，并且下界严格小于上界。
        # 这是稳定记忆衰减所需的基本约束。
        if not 0 < delta_min < delta_max < float("inf"):
            raise ValueError("delta bounds must satisfy 0 < delta_min < delta_max < infinity")

        # 防止配置文件中误写一个未实现的 delta 模式。
        if delta_mode not in {"vanilla", "scale_only", "scale_region", "scale_region_history"}:
            raise ValueError(f"unsupported delta mode: {delta_mode}")

        # 保存模型维度与运行配置。
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.num_scales = int(num_scales)
        self.num_regions = len(REGION_NAMES)
        self.delta_min = float(delta_min)
        self.delta_max = float(delta_max)
        self.delta_mode = delta_mode
        self.history_gamma = float(history_gamma)
        self.backend = backend

        # input_proj：
        # 把输入 x 投影成真正送入 selective scan 的输入 u。
        # 输入/输出最后一维均为 D，所以不会改变整体张量形状。
        self.input_proj = nn.Linear(d_model, d_model)

        # input_delta：
        # 根据每个时间步的输入特征生成 delta logits。
        # 它生成的是尚未经过 Sigmoid 和上下界映射的原始分数。
        self.input_delta = nn.Linear(d_model, d_model)

        # history_proj：
        # 把整段历史的池化表示转换成对 delta logits 的修正量。
        # 仅在 delta_mode == "scale_region_history" 时使用。
        self.history_proj = nn.Linear(d_model, d_model)

        # B_proj：
        # 根据输入动态生成 selective scan 中负责“写入状态”的 B_t。
        #
        # [B,L,N,S,D] -> [B,L,N,S,d_state]
        #
        # bias=False 表示这里只学习线性映射权重，不额外加入常数偏置。
        self.B_proj = nn.Linear(d_model, d_state, bias=False)

        # C_proj：
        # 根据输入动态生成 selective scan 中负责“读取状态”的 C_t。
        #
        # [B,L,N,S,D] -> [B,L,N,S,d_state]
        self.C_proj = nn.Linear(d_model, d_state, bias=False)

        # out_proj：
        # scan 完成后，在隐藏特征维 D 上再做一次线性变换。
        self.out_proj = nn.Linear(d_model, d_model)

        # norm：
        # 对“scan 输出 + 输入残差”做 LayerNorm。
        # 只归一化最后一个隐藏特征维 D。
        self.norm = nn.LayerNorm(d_model)

        # 每个尺度、每个隐藏通道都有一个可学习的 delta 偏置：
        # [S,D]
        #
        # 初始化为 0，表示训练开始时不主动偏向任何尺度。
        self.scale_bias = nn.Parameter(torch.zeros(num_scales, d_model))

        # 每个“分位区域 × 尺度 × 隐藏通道”都有一个可学习偏置：
        # [R,S,D]
        region_scale_bias = torch.zeros(self.num_regions, num_scales, d_model)

        # 初始化时：
        # median 区域偏置 = 0
        # upper 区域偏置 = -0.2
        # extreme 区域偏置 = -0.5
        #
        # 负偏置会使 Sigmoid 输出倾向于更小，因而使 delta 倾向于更小。
        # 在 A < 0 时，较小 delta 通常意味着状态衰减更慢、记忆更长。
        #
        # 这只是初始化先验，不是硬约束；训练后偏置可以改变。
        region_scale_bias[1].fill_(-0.2)
        region_scale_bias[2].fill_(-0.5)
        self.region_scale_bias = nn.Parameter(region_scale_bias)

        # 为每个隐藏通道初始化 d_state 个连续时间衰减速率。
        #
        # initial_A = [1, 2, ..., d_state]
        # A_log 初始化为 log([1, 2, ..., d_state])，并复制给每个隐藏通道：
        # [D,d_state]
        #
        # 真正参与 scan 的 A 会在下面的 A 属性中变成：
        # A = -exp(A_log)
        initial_A = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(initial_A.log().repeat(d_model, 1))

        # D 是 selective scan 输出公式 y_t = C_t h_t + D*u_t
        # 中的直接输入/跳跃连接系数。
        #
        # 每个隐藏通道一个可学习值，形状为 [D]，初始值均为 1。
        self.D = nn.Parameter(torch.ones(d_model))

    @property
    def A(self):
        """
        根据可学习的 A_log 构造真正的连续时间状态衰减参数 A。

        exp(A_log) 始终为正，再取负号后：

            A = -exp(A_log) < 0

        因此在 delta > 0 时：

            exp(delta * A) 位于 (0, 1)

        这使旧状态随时间衰减，而不是无界放大。

        A 的形状为 [D,d_state]。
        """
        return -torch.exp(self.A_log.float())

    def _mask(self, history_mask, x):
        """
        把可选的历史掩码统一整理为 [B,L,N,1]。

        掩码通常用于标记某个站点在某个历史时刻是否存在有效观测。
        本函数只服务于历史池化，不会直接屏蔽 selective scan 的输入。
        """

        # 没有传掩码时，默认所有历史位置均有效。
        #
        # x.shape[:3] 为 [B,L,N]，最后补 1 得到 [B,L,N,1]。
        if history_mask is None:
            return x.new_ones(*x.shape[:3], 1)

        # 接受两种常见形状：
        # [B,L,N,1] 或 [B,L,N,F]
        #
        # 只严格检查前 3 维是否与 x 一致。
        if history_mask.ndim != 4 or history_mask.shape[:3] != x.shape[:3]:
            raise ValueError("history_mask must have shape [B,L,N,1] or [B,L,N,F]")

        # 如果掩码最后一维包含多个原始特征 F，则只要其中任一特征有效，
        # 这个站点/时间位置就被视为有效。
        #
        # 输出统一为 [B,L,N,1]，并转换成与 x 相同的 dtype。
        return history_mask.to(dtype=x.dtype).amax(dim=-1, keepdim=True)

    def _delta(self, x, history_mask):
        """
        生成有严格上下界的 delta。

        输入：
            x: [B,L,N,S,D]

        输出：
            delta: [B,L,N,R,S,D]

        与标准 selective scan 相比，这里的创新不是“新增 delta”。
        标准 scan 本来就包含 delta；本模块重新设计的是 delta 的生成方式，
        使其同时受到输入、尺度、分位区域和历史状态的条件控制。
        """
        batch, length, stations, scales, channels = x.shape

        # 先根据当前输入生成基础 delta logits：
        #
        # input_delta(x):
        # [B,L,N,S,D]
        #
        # unsqueeze(3):
        # [B,L,N,1,S,D]
        #
        # expand 到三个分位区域：
        # [B,L,N,R,S,D]
        #
        # 此时三个区域暂时共享同一份输入条件 logits。
        logits = self.input_delta(x).unsqueeze(3).expand(
            -1, -1, -1, self.num_regions, -1, -1
        )

        # scale_only、scale_region、scale_region_history：
        # 为不同尺度加入可学习偏置。
        #
        # scale_bias:
        # [S,D] -> [1,1,1,1,S,D]
        #
        # 通过广播加到所有 batch、时间、站点和分位区域上。
        if self.delta_mode in {"scale_only", "scale_region", "scale_region_history"}:
            logits = logits + self.scale_bias[None, None, None, None]

        # scale_region、scale_region_history：
        # 再加入“区域 × 尺度”联合偏置。
        #
        # region_scale_bias:
        # [R,S,D] -> [1,1,1,R,S,D]
        #
        # 从这一步开始，不同分位区域可以具有不同的 delta。
        if self.delta_mode in {"scale_region", "scale_region_history"}:
            logits = logits + self.region_scale_bias[None, None, None]

        # scale_region_history：
        # 根据整个历史窗口的汇总特征，对当前样本的 delta 再做一次自适应修正。
        if self.delta_mode == "scale_region_history":

            # _mask(...) 得到 [B,L,N,1]；
            # 再插入尺度前的单例维，得到 [B,L,N,1,1]。
            #
            # 它可以广播到 x 的 [B,L,N,S,D]。
            mask = self._mask(history_mask, x).unsqueeze(3)

            # 对时间维 L 做带掩码平均：
            #
            # 分子：
            # (x * mask).sum(dim=1) -> [B,N,S,D]
            #
            # 分母：
            # mask.sum(dim=1) -> [B,N,1,1]
            #
            # clamp_min(1.0) 防止某个站点所有历史位置都无效时除以 0。
            pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

            # history_proj 后仍是 [B,N,S,D]。
            # tanh 把修正值限制在 [-1,1]，避免历史修正项无限增大。
            correction = torch.tanh(self.history_proj(pooled))

            # correction[:, None, :, None]：
            # [B,N,S,D] -> [B,1,N,1,S,D]
            #
            # 因而同一个样本、站点和尺度上的历史修正：
            # - 对所有时间步共享；
            # - 对三个分位区域共享；
            # - 但仍可在不同隐藏通道上不同。
            #
            # 区域之间的差异主要来自上面的 region_scale_bias。
            logits = logits + self.history_gamma * correction[:, None, :, None]

        # 使用 Sigmoid 把任意 logits 映射到 (0,1)，再线性映射到：
        #
        # (delta_min, delta_max)
        #
        # 数学形式：
        #
        # delta = delta_min
        #       + (delta_max - delta_min) * sigmoid(logits)
        #
        # 因此传入 scan 的 delta 始终为正且有上界。
        return self.delta_min + (self.delta_max - self.delta_min) * torch.sigmoid(logits)

    def _summaries(self, delta):
        """
        根据当前批次的 delta 和 A，计算便于记录的记忆诊断量。

        这些 summary 不参与 y 的计算，只用于观察模型学到了怎样的
        delta 和近似半衰期。
        """

        # exp_A = -A > 0，形状为 [D,d_state]。
        # 它表示各通道、各状态槽的正衰减速率。
        exp_A = torch.exp(self.A_log.float())

        # 对每个隐藏通道，把 1 / exp_A 在 d_state 个状态槽上取平均：
        # [D,d_state] -> [D]
        #
        # 该量用于汇总不同状态槽对应的半衰期。
        reciprocal_A_mean = torch.exp(-self.A_log.float()).mean(dim=-1)

        # 单个状态模式的半衰期来自：
        #
        # exp(-delta * exp_A * t_half) = 1/2
        #
        # 因而：
        #
        # t_half = ln(2) / (delta * exp_A)
        #
        # 这里先对每个通道的 1/exp_A 做状态维平均，再对整个 delta
        # 张量求平均，得到一个全局平均半衰期。
        half_life_mean = (
            math.log(2.0) * (delta.reciprocal() * reciprocal_A_mean).mean()
        )

        # 除分位区域 R 外，对 batch、时间、站点、尺度、通道全部取平均。
        #
        # delta_by_region.shape == [R]
        delta_by_region = delta.mean(dim=(0, 1, 2, 4, 5))

        # 各分位区域的平均半衰期，形状同样为 [R]。
        half_by_region = math.log(2.0) * (
            delta.reciprocal() * reciprocal_A_mean
        ).mean(dim=(0, 1, 2, 4, 5))

        # 汇总当前批次中 delta 的最小值、最大值、均值和各区域均值。
        delta_summary = {
            "min": delta.amin(),
            "max": delta.amax(),
            "mean": delta.mean(),
            "by_region": delta_by_region,
        }

        # 半衰期与 delta、exp_A 均成反比：
        #
        # 最短半衰期：
        # 最大 delta × 最大 exp_A
        #
        # 最长半衰期：
        # 最小 delta × 最小 exp_A
        half_life_summary = {
            "min": delta.new_tensor(math.log(2.0)) / (delta.amax() * exp_A.max()),
            "max": delta.new_tensor(math.log(2.0)) / (delta.amin() * exp_A.min()),
            "mean": half_life_mean,
            "by_region": half_by_region,
        }

        return delta_summary, half_life_summary

    def forward(self, x, history_mask=None, return_diagnostics=False):
        """
        执行分位区域条件、多尺度 selective scan。

        参数
        ----
        x:
            输入时序表示，形状 [B,L,N,S,D]。
        history_mask:
            可选历史有效性掩码，用于 _delta() 中的历史池化。
        return_diagnostics:
            是否额外返回完整 delta、A 和转移谱半径。

        返回
        ----
        y:
            scan 后的区域化多尺度表示，[B,L,N,R,S,D]。
        delta_summary:
            当前批次 delta 的汇总统计。
        half_life_summary:
            当前批次记忆半衰期的汇总统计。
        diagnostics:
            return_diagnostics=False 时为 None；
            否则包含完整 delta、A 和 transition_spectral_radius。
        """

        # 本模块只接受五维输入 [B,L,N,S,D]。
        if x.ndim != 5:
            raise ValueError(f"quantile SSM expects [B,L,N,S,D], got {tuple(x.shape)}")

        batch, length, stations, scales, channels = x.shape

        # 检查输入中的尺度数 S 和隐藏维 D 是否与初始化配置一致。
        if scales != self.num_scales or channels != self.d_model:
            raise ValueError(
                "quantile SSM scale or channel dimension does not match its configuration"
            )

        # 生成 scan 的输入 u：
        # [B,L,N,S,D] -> [B,L,N,S,D]
        u = self.input_proj(x)

        # 生成带分位区域维的有界 delta：
        # [B,L,N,R,S,D]
        delta = self._delta(x, history_mask)

        # 根据当前输入动态生成 B_t 和 C_t：
        #
        # dynamic_B: [B,L,N,S,d_state]
        # dynamic_C: [B,L,N,S,d_state]
        #
        # tanh 将它们限制在 [-1,1]。
        # B_t 控制当前输入如何写入状态，C_t 控制如何从状态中读取。
        dynamic_B = torch.tanh(self.B_proj(x))
        dynamic_C = torch.tanh(self.C_proj(x))

        # selective scan 接口把每条独立序列视为一个 scan batch。
        #
        # 这里每个：
        # 原始样本 × 站点 × 分位区域 × 时间尺度
        #
        # 都是一条独立扫描序列，因此总扫描实例数为：
        scan_batch = batch * stations * self.num_regions * scales

        # -------------------- 整理 u 的维度 --------------------
        #
        # 原始 u：
        # [B,L,N,S,D]
        #
        # 加入并复制区域维：
        # [B,L,N,R,S,D]
        #
        # 调整顺序：
        # [B,N,R,S,D,L]
        #
        # 展平 B、N、R、S：
        # [B*N*R*S,D,L]
        #
        # 这是 bounded_selective_scan 所要求的 u 布局：
        # [scan_batch,channels,length]
        u_scan = (
            u.unsqueeze(3)
            .expand(-1, -1, -1, self.num_regions, -1, -1)
            .permute(0, 2, 3, 4, 5, 1)
            .reshape(scan_batch, channels, length)
        )

        # ------------------ 整理 delta 的维度 ------------------
        #
        # delta 已包含区域维：
        # [B,L,N,R,S,D]
        #
        # 调整并展平后：
        # [B*N*R*S,D,L]
        #
        # 与 u_scan 的形状完全相同。
        delta_scan = (
            delta.permute(0, 2, 3, 4, 5, 1)
            .reshape(scan_batch, channels, length)
        )

        # -------------------- 整理 B 的维度 --------------------
        #
        # dynamic_B:
        # [B,L,N,S,d_state]
        #
        # B 当前没有单独的区域条件，所以先在三个区域间复制：
        # [B,L,N,R,S,d_state]
        #
        # 调整并展平后：
        # [B*N*R*S,d_state,L]
        #
        # 注意：
        # 本模块的三个区域主要通过不同 delta 形成不同记忆速度；
        # 同一输入下，它们目前共享 u、B 和 C。
        B_scan = (
            dynamic_B.unsqueeze(3)
            .expand(-1, -1, -1, self.num_regions, -1, -1)
            .permute(0, 2, 3, 4, 5, 1)
            .reshape(scan_batch, self.d_state, length)
        )

        # C 的整理过程与 B 完全相同：
        # [B,L,N,S,d_state]
        # -> [B,L,N,R,S,d_state]
        # -> [B*N*R*S,d_state,L]
        C_scan = (
            dynamic_C.unsqueeze(3)
            .expand(-1, -1, -1, self.num_regions, -1, -1)
            .permute(0, 2, 3, 4, 5, 1)
            .reshape(scan_batch, self.d_state, length)
        )

        # 执行 selective scan。
        #
        # scan 内部的核心递推仍是标准形式：
        #
        # h_t = exp(delta_t * A) * h_{t-1} + delta_t * B_t * u_t
        # y_t = C_t * h_t + D * u_t
        #
        # 当前模块没有修改这条递推公式，主要改变的是：
        # - delta 如何由尺度、分位区域和历史信息生成；
        # - 如何把区域和尺度展开成多条独立 scan 序列。
        #
        # 输出形状：
        # [scan_batch,D,L]
        y = bounded_selective_scan(
            u_scan,
            delta_scan,
            self.A.to(dtype=u.dtype, device=u.device),
            B_scan,
            C_scan,
            self.D.to(dtype=u.dtype),
            self.backend,
        )

        # 把 scan_batch 重新拆回：
        #
        # [B*N*R*S,D,L]
        # -> [B,N,R,S,D,L]
        # -> [B,L,N,R,S,D]
        y = y.reshape(
            batch, stations, self.num_regions, scales, channels, length
        ).permute(0, 5, 1, 2, 3, 4)

        # u 原本没有区域维，因此复制到三个区域：
        # [B,L,N,S,D] -> [B,L,N,R,S,D]
        residual = u.unsqueeze(3).expand_as(y)

        # 对 scan 输出做线性投影，加上输入残差，再做 LayerNorm：
        #
        # output = LayerNorm(u + Linear(scan_output))
        #
        # 输出形状保持 [B,L,N,R,S,D]。
        y = self.norm(residual + self.out_proj(y))

        # 生成便于训练日志和分析使用的汇总诊断。
        # 这些统计不会反过来改变 y。
        delta_summary, half_life_summary = self._summaries(delta)

        # 默认不返回完整诊断张量，以节省显存和日志空间。
        diagnostics = None

        if return_diagnostics:
            # exp(A_log) 为正衰减速率：
            # [D,d_state]
            #
            # 对每个通道取最小衰减速率：
            # [D]
            #
            # 因为 exp(-delta * rate) 随 rate 增大而减小，
            # 最小 rate 对应状态转移矩阵中最大的特征值模，
            # 即这里的 transition spectral radius。
            min_decay = torch.exp(self.A_log.float()).amin(dim=-1).to(delta.dtype)

            diagnostics = {
                # 完整有界 delta：
                # [B,L,N,R,S,D]
                "delta": delta,

                # 连续时间衰减参数：
                # [D,d_state]，所有元素严格小于 0。
                "A": self.A,

                # 每个位置、区域、尺度、通道上的最大状态保留系数：
                # [B,L,N,R,S,D]
                #
                # 数值位于 (0,1)：
                # - 越接近 1：衰减越慢，记忆越长；
                # - 越接近 0：衰减越快，记忆越短。
                "transition_spectral_radius": torch.exp(
                    -delta * min_decay[None, None, None, None, None]
                ),
            }

        # y 是真正供后续网络使用的特征。
        # summaries 和 diagnostics 主要用于检查模型记忆行为。
        return y, delta_summary, half_life_summary, diagnostics
