import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 本文件包含两个空间混合模块：
# 1. HistoryConditionedDirectedSpatialMixer：
#    使用历史原始风场、站点坐标、固定候选邻居和节点自适应关系，构造有向空间权重。
# 2. STM3AGCCNBaselineMixer：
#    保留原始 STM3 的自适应图卷积，作为对照基线。
#
# 常用张量维度记号：
# B：batch size
# L：历史序列长度
# N：站点数
# S：时间尺度数
# D：隐藏特征维度（d_model）
# K：每个目标站点给定的邻居数
# E：实际候选来源数。第一个模块会额外加入目标站点自身，因此 E = K + 1
# ---------------------------------------------------------------------------


def _gather_stations(x, index):
    # x 的前三维应为 [B,L,N]，index 的形状为 [N,E]。
    # 对每个目标站点 n，取出 index[n] 指定的 E 个来源站点。
    #
    # 典型输出：
    # x=[B,L,N,2]   -> [B,L,N,E,2]
    # x=[B,L,N,S,D] -> [B,L,N,E,S,D]
    #
    # 输出中的第 3 维 N 仍表示“目标站点”，新增的第 4 维 E 表示该目标的候选来源。
    return x[:, :, index]


def _log_parameter(value, name):
    # beta 系数必须为正。这里优化的是 log(beta)，实际使用时再取 exp，
    # 从参数化方式上保证训练过程中 beta 始终大于 0。
    value = float(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return nn.Parameter(torch.tensor(math.log(value)))


class HistoryConditionedDirectedSpatialMixer(nn.Module):
    # 基于历史条件的有向空间混合器。
    #
    # 对每个“目标站点—候选来源站点”边，空间注意力 logit 最多由四部分组成：
    #
    # content + beta_geo * geo + beta_wind * wind + beta_adaptive * adaptive
    #
    # - content：当前隐藏状态的 Query-Key 内容相关性；
    # - geo：随站点距离增大而减小的地理先验；
    # - wind：来源站点风速、风向对齐度和距离衰减共同形成的有向传播分数；
    # - adaptive：由目标/来源节点嵌入学习到的非负关系偏置。
    #
    # softmax 沿候选来源维 E 进行，因此每个目标站点会从其 K 个邻居和自身中
    # 自适应聚合信息。
    def __init__(
        self,
        d_model,
        num_nodes,
        node_emb_dim,
        geo=True,
        wind=True,
        content=True,
        scale_conditioned=True,
        node_adaptive=True,
        geo_length_scale=2.0,
        wind_length_scale=2.0,
        beta_geo=1.0,
        beta_wind=1.0,
        beta_adaptive=1.0,
        chunk_size=32,
    ):
        super().__init__()

        # 固定结构信息及各空间分量的开关。
        self.num_nodes = int(num_nodes)
        self.geo_enabled = bool(geo)
        self.wind_enabled = bool(wind)
        self.content_enabled = bool(content)

        # True：每个时间尺度分别计算 content 权重，logits 形状为 [B,L,N,S,E]。
        # False：先对尺度维取平均，再得到所有尺度共享的空间权重 [B,L,N,E]。
        self.scale_conditioned = bool(scale_conditioned)
        self.node_adaptive_enabled = bool(node_adaptive)

        # 两个长度尺度与 coords 使用相同单位。当前输入合同规定 coords 为
        # [east_km, north_km]，因此这里的单位也是 km。
        # 它们是固定超参数，不是 nn.Parameter。
        self.geo_length_scale = float(geo_length_scale)
        self.wind_length_scale = float(wind_length_scale)

        # 沿时间维分块计算，主要用于控制中间 attention 张量的显存占用。
        # 分块只改变计算方式，不改变各时间步之间的数学关系。
        self.chunk_size = int(chunk_size)

        # 提前拒绝无效配置，避免后续出现无意义的距离衰减或空维度。
        if self.num_nodes <= 0 or int(node_emb_dim) <= 0:
            raise ValueError("num_nodes and node_emb_dim must be positive")
        if self.geo_length_scale <= 0 or self.wind_length_scale <= 0:
            raise ValueError("spatial length scales must be positive")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        # 三个 beta 的底层参数均位于对数空间；通过对应 property 取得正的有效值。
        self.log_beta_geo = _log_parameter(beta_geo, "beta_geo")
        self.log_beta_wind = _log_parameter(beta_wind, "beta_wind")
        self.log_beta_adaptive = _log_parameter(beta_adaptive, "beta_adaptive")

        # 内容注意力的线性投影。输入和输出最后一维均为 D。
        self.query = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

        # 空间聚合结果以残差方式写回。sigmoid 将门值限制在 (0,1)。
        # 初始 raw 值为 0，因此初始有效门值为 sigmoid(0)=0.5。
        self.residual_gate = nn.Parameter(torch.tensor(0.0))

        if self.node_adaptive_enabled:
            # 分别为“目标角色”和“来源角色”学习节点嵌入。
            # 两套嵌入不共享，使 target<-source 的关系可以是有向的。
            self.target_embeddings = nn.Parameter(
                torch.empty(self.num_nodes, node_emb_dim)
            )
            self.source_embeddings = nn.Parameter(
                torch.empty(self.num_nodes, node_emb_dim)
            )
            nn.init.xavier_uniform_(self.target_embeddings)
            nn.init.xavier_uniform_(self.source_embeddings)

    @property
    def beta_geo(self):
        # 有效地理系数始终为正。
        return self.log_beta_geo.exp()

    @property
    def beta_wind(self):
        # 有效风场系数始终为正。
        return self.log_beta_wind.exp()

    @property
    def beta_adaptive(self):
        # 有效节点自适应系数始终为正。
        return self.log_beta_adaptive.exp()

    def _validate(self, x, history_wind_uv, coords, neighbor_index):
        # x 是多尺度历史隐藏状态，固定形状为 [B,L,N,S,D]。
        if x.ndim != 5:
            raise ValueError(f"spatial mixer expects [B,L,N,S,D], got {tuple(x.shape)}")
        stations = x.shape[2]
        if stations != self.num_nodes:
            raise ValueError(
                f"spatial mixer expects N={self.num_nodes}, got {stations}"
            )

        # neighbor_index[n,k] 是目标站点 n 的第 k 个“来源站点”索引，
        # 输入形状为 [N,K]。自身不应提前放入，因为 forward 会统一追加一次 self。
        if neighbor_index is None:
            raise ValueError("history_directed spatial mode requires neighbor_index")
        if not torch.is_tensor(neighbor_index):
            raise TypeError("neighbor_index must be a torch.Tensor")

        # 必须由上游明确提供整数索引，不能把浮点索引静默转换为 long。
        if neighbor_index.dtype != torch.long:
            raise ValueError("neighbor_index must have torch.long integer dtype")
        if neighbor_index.ndim != 2 or neighbor_index.shape[0] != stations:
            raise ValueError("neighbor_index must have shape [N,K]")
        if neighbor_index.numel() and (
            neighbor_index.min() < 0 or neighbor_index.max() >= stations
        ):
            raise ValueError("neighbor_index contains an out-of-range station")
        if neighbor_index.shape[1]:
            self_index = torch.arange(stations, device=neighbor_index.device)[:, None]

            # 候选邻居中不能包含自身，也不能在同一行重复同一来源。
            if (neighbor_index == self_index).any():
                raise ValueError("neighbor_index must exclude self; self is appended once")
            sorted_neighbors = neighbor_index.sort(dim=1).values
            if (sorted_neighbors[:, 1:] == sorted_neighbors[:, :-1]).any():
                raise ValueError("neighbor_index rows must not contain duplicate sources")

        # 地理项和风场项都依赖站点之间的空间方向/距离，因此任一启用时都需要坐标。
        if (self.geo_enabled or self.wind_enabled) and coords is None:
            raise ValueError("geo or wind spatial edges require coords")
        if coords is not None and not torch.is_tensor(coords):
            raise TypeError("coords must be a torch.Tensor")
        if coords is not None and coords.shape != (stations, 2):
            raise ValueError("coords must have shape [N,2]")

        # coords[:,0] 必须是向东距离 east_km，coords[:,1] 必须是向北距离 north_km。
        # 这里不接受原始经纬度或含 NaN/Inf 的坐标。
        if coords is not None and (
            not torch.is_floating_point(coords) or not torch.isfinite(coords).all()
        ):
            raise ValueError("coords must contain finite floating-point east/north km")

        if self.wind_enabled:
            # history_wind_uv 与 x 使用完全相同的 B、L、N，只在最后保存两个物理分量：
            # [...,0] = u_east，向东为正；
            # [...,1] = v_north，向北为正。
            # 该输入应为历史窗口中的原始物理风场，而不是 Z-score 后的 history_met。
            if history_wind_uv is None:
                raise ValueError("wind spatial edges require history_wind_uv")
            if not torch.is_tensor(history_wind_uv):
                raise TypeError("history_wind_uv must be a torch.Tensor")
            if history_wind_uv.shape != x.shape[:3] + (2,):
                raise ValueError("history_wind_uv must have shape [B,L,N,2]")
            if not torch.is_floating_point(history_wind_uv):
                raise ValueError("history_wind_uv must have floating-point dtype")
            if not torch.isfinite(history_wind_uv).all():
                raise ValueError("history_wind_uv must contain finite physical wind")

    def _static_edge_terms(self, coords, sources):
        # coords: [N,2]；sources: [N,E]。
        # target_coords[n,0] 是目标站点 n 的坐标，
        # source_coords[n,e] 是该目标的第 e 个候选来源坐标。
        target_coords = coords[:, None, :]
        source_coords = coords[sources]

        # vector = target - source，方向明确表示“从来源站点指向目标站点”。
        # 这与后续判断来源地风是否吹向目标站点相一致。
        vector = target_coords - source_coords
        distance = torch.linalg.vector_norm(vector, dim=-1)

        # direction 为单位边方向 [N,E,2]。
        # self edge 距离为 0；clamp 只防止除零，其方向向量仍为 [0,0]。
        direction = vector / distance.clamp_min(1e-8).unsqueeze(-1)

        # 地理项是非正 logit 偏置：距离越远，惩罚绝对值越大；self edge 为 0。
        geo = -distance / self.geo_length_scale

        # 风传播随距离按指数衰减，范围为 (0,1]；self edge 的衰减值为 1。
        # 不过 self edge 的 direction 为零，所以最终风向对齐分数仍为 0。
        decay = torch.exp(-distance / self.wind_length_scale)
        return geo, decay, direction

    def _adaptive_edge_bias(self, sources):
        # 为每个目标站点与其 E 个候选来源计算角色相关的嵌入内积。
        # target: [N,1,H]，source: [N,E,H]，输出为 [N,E]。
        target = self.target_embeddings[:, None]
        source = self.source_embeddings[sources]
        scale = math.sqrt(self.target_embeddings.shape[-1])

        # 除以 sqrt(H) 控制内积量级；ReLU 使该先验只提供非负增强。
        return F.relu((target * source).sum(dim=-1) / scale)

    def _wind_edge_score(self, wind_uv, edge_direction, wind_decay, sources):
        # wind_uv: [B,T,N,2]，T 是当前时间块长度。
        # 从每个候选来源站点提取其风矢量，得到 [B,T,N,E,2]。
        source_wind = _gather_stations(wind_uv, sources)

        # speed 是原始物理风速；unit_wind 只保留风的“吹向”方向。
        speed = torch.linalg.vector_norm(source_wind, dim=-1)
        unit_wind = source_wind / speed.clamp_min(1e-8).unsqueeze(-1)

        # 风方向与“来源->目标”边方向做点积：
        # 正值表示风在推动来源站点的信息朝目标站点传播；
        # 负值表示反向风，经 ReLU 后不产生正向传播贡献。
        # edge_direction 增加两个前导维，以便在 B 和 T 上广播。
        alignment = F.relu(
            (unit_wind * edge_direction[None, None]).sum(dim=-1)
        )

        # 最终风场分数 = 来源地风速 × 正向对齐度 × 空间距离衰减。
        # 输出形状为 [B,T,N,E]。self edge 因方向为零，其分数为零。
        return speed * alignment * wind_decay[None, None]

    def forward(
        self, x, history_wind_uv, coords, neighbor_index, return_diagnostics=False
    ):
        # 输入合同：
        # x:               [B,L,N,S,D]，多尺度历史隐藏状态
        # history_wind_uv: [B,L,N,2]，历史原始物理 u/v 风场
        # coords:          [N,2]，按 [east_km,north_km] 排列的站点坐标
        # neighbor_index:  [N,K]，每个目标站点的 K 个候选来源，不含 self
        self._validate(x, history_wind_uv, coords, neighbor_index)
        batch, length, stations, scales, channels = x.shape
        device = x.device

        # 验证通过后只对齐 device。neighbor_index 不转换 dtype，从而保留严格整数合同。
        neighbor_index = neighbor_index.to(device=device)

        # 风场保持原始物理数值，仅对齐计算 device 和浮点 dtype；
        # 这里没有执行归一化或标准化。
        history_wind_uv = (
            history_wind_uv.to(device=device, dtype=x.dtype)
            if history_wind_uv is not None
            else None
        )

        # 在每行邻居列表最后追加一次目标站点自身：
        # neighbor_index [N,K] -> sources [N,E]，其中 E=K+1。
        # 后续 weights[..., -1] 因而固定表示 self weight。
        self_index = torch.arange(stations, device=device)[:, None]
        sources = torch.cat((neighbor_index, self_index), dim=1)

        # 坐标也只对齐 device/dtype，不改变其 east/north km 语义。
        coords = coords.to(device=device, dtype=x.dtype) if coords is not None else None

        # 距离、边方向和距离衰减只由固定坐标/候选边决定，
        # 与 batch、时间和尺度无关，因此在时间分块循环外只计算一次。
        if coords is not None:
            geo, wind_decay, edge_direction = self._static_edge_terms(coords, sources)
        else:
            geo = wind_decay = edge_direction = None

        # 节点自适应项同样是跨 batch、时间和尺度共享的 [N,E] 先验。
        adaptive = (
            self._adaptive_edge_bias(sources)
            if self.node_adaptive_enabled
            else None
        )

        outputs = []

        # 只有请求完整诊断时才保存所有时间块的权重/风分数，
        # 避免正常训练过程中保留大体积中间张量。
        all_weights = [] if return_diagnostics else None
        all_wind_scores = [] if return_diagnostics and self.wind_enabled else None

        # 以下累计量用于构造轻量 summary，无论是否返回完整 diagnostics 都会计算。
        weight_sum = x.new_zeros(())
        entropy_sum = x.new_zeros(())
        self_sum = x.new_zeros(())
        count = 0

        # 沿历史时间维 L 分块。每个块独立计算对应时间步的空间聚合，
        # 最后再沿时间维拼回完整输出。
        for start in range(0, length, self.chunk_size):
            stop = min(start + self.chunk_size, length)
            chunk = x[:, start:stop]
            wind_score = None

            if self.wind_enabled:
                # 当前块的风场分数形状为 [B,T,N,E]，T=stop-start。
                wind_score = self._wind_edge_score(
                    history_wind_uv[:, start:stop],
                    edge_direction,
                    wind_decay,
                    sources,
                )

            if self.scale_conditioned:
                # 每个尺度拥有独立的一组候选来源 logits：
                # [B,T,N,S,E]。
                logits = x.new_zeros(
                    batch, stop - start, stations, scales, sources.shape[1]
                )

                if self.content_enabled:
                    # chunk 经投影后仍为 [B,T,N,S,D]。
                    # query 在来源位置插入长度为 1 的维度：
                    # [B,T,N,1,S,D]。
                    query = self.query(chunk).unsqueeze(3)

                    # 为每个目标站点收集 E 个来源的 key：
                    # [B,T,N,E,S,D]。
                    keys = _gather_stations(self.key(chunk), sources)

                    # 在 D 上做缩放点积，先得到 [B,T,N,E,S]，
                    # 再交换 E/S，统一为 logits 所需的 [B,T,N,S,E]。
                    content = (query * keys).sum(dim=-1) / math.sqrt(channels)
                    logits = logits + content.transpose(3, 4)

                if self.geo_enabled:
                    # geo 为 [N,E]，在 B、T、S 上广播。
                    # 地理先验跨尺度共享，但最终权重仍会因 content 不同而随尺度变化。
                    logits = logits + self.beta_geo * geo[None, None, :, None]

                if self.wind_enabled:
                    # wind_score 为 [B,T,N,E]，插入尺度维后在 S 上共享。
                    logits = logits + self.beta_wind * wind_score.unsqueeze(3)

                if adaptive is not None:
                    # adaptive 为 [N,E]，在 B、T、S 上广播。
                    logits = (
                        logits
                        + self.beta_adaptive * adaptive[None, None, :, None]
                    )
            else:
                # 关闭尺度条件后，先对 S 取平均得到 [B,T,N,D]，
                # 再为所有尺度计算一组共享空间权重。
                pooled = chunk.mean(dim=3)
                logits = x.new_zeros(
                    batch, stop - start, stations, sources.shape[1]
                )

                if self.content_enabled:
                    # query: [B,T,N,1,D]；keys: [B,T,N,E,D]；
                    # 缩放点积结果为 [B,T,N,E]。
                    query = self.query(pooled).unsqueeze(3)
                    keys = _gather_stations(self.key(pooled), sources)
                    logits = logits + (query * keys).sum(dim=-1) / math.sqrt(channels)

                if self.geo_enabled:
                    # [N,E] 在 B、T 上广播。
                    logits = logits + self.beta_geo * geo[None, None]

                if self.wind_enabled:
                    # wind_score 已是 [B,T,N,E]。
                    logits = logits + self.beta_wind * wind_score

                if adaptive is not None:
                    # adaptive [N,E] 在 B、T 上广播。
                    logits = logits + self.beta_adaptive * adaptive[None, None]

            # 始终沿最后的候选来源维 E 归一化。
            # 对每个目标站点（以及启用时的每个尺度），E 个权重之和为 1。
            weights = F.softmax(logits, dim=-1)

            # 为每个目标站点收集 E 个来源的 value：
            # [B,T,N,E,S,D]。
            values = _gather_stations(self.value(chunk), sources)

            if self.scale_conditioned:
                # weights: [B,T,N,S,E] -> [B,T,N,E,S,1]，
                # 与 values 逐元素相乘，再沿来源维 E（dim=3）求和。
                # mixed 最终恢复为 [B,T,N,S,D]。
                mixed = (weights.transpose(3, 4).unsqueeze(-1) * values).sum(dim=3)
            else:
                # 一组共享 weights [B,T,N,E] 同时作用于所有 S 个尺度和 D 个通道。
                mixed = (weights[..., None, None] * values).sum(dim=3)

            # 以可学习门控的残差形式写回空间信息。
            gate = torch.sigmoid(self.residual_gate)
            outputs.append(chunk + gate * self.out(mixed))

            # 累加空间权重的归一化和熵统计。
            # 因 self 被放在 sources 最后一列，所以 weights[..., -1] 是 self 权重。
            weight_sum = weight_sum + weights.sum()
            entropy_sum = entropy_sum - (weights * weights.clamp_min(1e-12).log()).sum()
            self_sum = self_sum + weights[..., -1].sum()
            count += weights.numel()

            if return_diagnostics:
                all_weights.append(weights)
                if all_wind_scores is not None:
                    all_wind_scores.append(wind_score)

        # count 统计所有单独权重元素；除以 E 后得到独立 softmax 分布的数量。
        # scale_conditioned=True 时，targets_seen=B*L*N*S；
        # scale_conditioned=False 时，targets_seen=B*L*N。
        edge_count = sources.shape[1]
        targets_seen = count // edge_count

        # sum_mean 理论上应接近 1；
        # entropy_mean 越低表示候选来源选择越尖锐；
        # self_weight_mean 表示平均保留多少目标站点自身信息。
        summary = {
            "sum_mean": weight_sum / targets_seen,
            "entropy_mean": entropy_sum / targets_seen,
            "self_weight_mean": self_sum / targets_seen,
        }

        diagnostics = None
        if return_diagnostics:
            # 各时间块沿 dim=1 拼接，恢复完整历史长度 L。
            diagnostics = {
                "spatial_weights": torch.cat(all_weights, dim=1),
                "spatial_sources": sources,
            }
            if all_wind_scores is not None:
                diagnostics["spatial_wind_score"] = torch.cat(
                    all_wind_scores, dim=1
                )

        # 主输出形状与 x 完全一致，均为 [B,L,N,S,D]。
        return torch.cat(outputs, dim=1), summary, diagnostics
