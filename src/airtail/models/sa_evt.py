"""Shape-adaptive EVT with an I-spline bulk."""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


from airtail.models.ispline import ISplineBulkQuantile
from airtail.models.evt_tail import PenultimateShapeAdaptiveEVTTail


@dataclass
class QuantileHeadOutput:
    """
    统一封装不同分位数头模式的输出。

    使用 dataclass 后，调用方可以通过字段名访问结果，而不需要记忆
    普通 tuple 中每个位置代表什么。

    典型形状
    --------
    quantiles:
        最终请求的全部分位数预测，[B,H,N,Q]。

    median:
        tau=0.5 的中位数锚点，[B,H,N,1]。

    bulk_coefficients:
        bulk I-spline 使用的正系数，[B,H,N,K]。
        direct 模式为了保持统一输出接口，也会生成这一字段，但它不参与
        direct 模式中二次分位函数的 quantiles 计算。

    threshold_u:
        分位阈值 tau_u 对应的预测值 q(tau_u)，[B,H,N,1]。

    threshold_slope:
        bulk 分位函数在 tau_u 附近的导数 q'(tau_u)，[B,H,N,1]。

    tail_sigma:
        tail 的尺度参数 sigma，[B,H,N,1]。

    tail_xi:
        tail 的形状参数 xi，[B,H,N,1]。

    注意：
    三种模式共用这一输出结构，因此在不实际使用 GPD tail 的模式下，
    tail_sigma 和 tail_xi 也会填入具有兼容含义的值。
    """

    quantiles: torch.Tensor
    median: torch.Tensor
    bulk_coefficients: torch.Tensor
    threshold_u: torch.Tensor
    threshold_slope: torch.Tensor
    tail_sigma: torch.Tensor
    tail_xi: torch.Tensor


def normalize_taus(taus, batch, device, dtype, tau_max):
    """
    将调用方提供的分位点统一整理为 [B,Q]，并完成合法性检查。

    参数
    ----
    taus:
        请求的分位水平，可以是：

        - [Q]：整个 batch 使用同一组分位点；
        - [B,Q]：每个样本可以提供自己的一组分位点。

    batch:
        当前 batch 大小 B。

    device, dtype:
        目标设备和数据类型。整理后的 taus 会与 horizon_states 保持一致。

    tau_max:
        配置允许查询的最大分位水平。

    返回
    ----
    taus:
        形状统一为 [B,Q]，并已放到正确 device、dtype 上。

    合法范围
    --------
        0.5 <= tau < 1.0
        tau <= tau_max

    本模型只处理从中位数开始的上半部分分位函数，因此不接受 tau<0.5。
    """

    if taus is None:
        raise ValueError("taus are required")

    if taus.ndim == 1:
        taus = taus[None].expand(batch, -1)

    elif taus.ndim != 2 or taus.shape[0] != batch:
        raise ValueError("taus must have shape [Q] or [B,Q]")

    taus = taus.to(device=device, dtype=dtype)

    if torch.any(taus < 0.5) or torch.any(taus >= 1.0):
        raise ValueError("taus must satisfy 0.5 <= tau < 1")

    if torch.any(taus > tau_max):
        raise ValueError(f"taus exceed configured tau_max={tau_max}")

    return taus


class _ReservedHeadParameters(nn.Module):
    """Preserve the state-dict slots and RNG consumption of the recorded model."""

    def __init__(self, d_model, num_basis, coefficient_epsilon, tau_u):
        """
        初始化直接非交叉分位数头。

        参数
        ----
        d_model:
            每类区域状态的隐藏特征维度 D。

        num_basis:
            为统一输出接口生成的 bulk 系数数量 K。

        coefficient_epsilon:
            正值下界。加在 softplus 输出上，使 slope 和 coefficients
            严格大于该下界，而不是只能保证非负。

        tau_u:
            bulk/tail 设计中的阈值分位点。direct 模式虽然不真正切换到
            GPD tail，仍会在该位置计算 threshold 和 threshold_slope，
            以便不同模式返回完全一致的字段。
        """
        super().__init__()

        self.median_proj = nn.Linear(d_model, 1)

        self.slope_proj = nn.Linear(d_model, 1)

        self.curvature_proj = nn.Linear(d_model, 1)

        self.coefficient_proj = nn.Linear(d_model, num_basis)

        self.coefficient_epsilon = float(coefficient_epsilon)
        self.tau_u = float(tau_u)



class UnifiedQuantileHead(nn.Module):
    """
    统一管理 direct、I-spline 和 I-spline+tail 三类分位数头。

    这个类本身主要承担两件事：

        1. 根据 config 创建各个候选分支；
        2. 在 forward 中按 mode 选择实际计算路径。

    无论选择哪种模式，返回值始终是 QuantileHeadOutput，方便训练、
    评估和诊断代码使用相同接口。
    """

    def __init__(self, d_model, config):
        """
        根据配置初始化统一分位数头。

        重要配置字段
        ------------
        mode:
            选择实际分位数头路径。

        tau_u:
            bulk 与 tail 的阈值分位水平。

        tau_max:
            允许调用方查询的最大分位水平。

        bulk_tau_max:
            I-spline bulk 定义域使用的上端分位水平。

        num_basis, spline_degree:
            I-spline 的基函数数量和次数。

        coefficient_epsilon:
            bulk 正系数与 direct slope 的正值下界。

        xi_min, xi_max:
            GPD 形状参数 xi 的配置范围。
        """
        super().__init__()

        self.mode = config["mode"]
        self.tau_u = float(config["tau_u"])
        self.tau_max = float(config["tau_max"])
        if self.mode != "psa_evt":
            raise ValueError("AirTail requires the SA-EVT quantile head")

        bulk_upper = config["bulk_tau_max"]
        if float(bulk_upper) != self.tau_u:
            raise ValueError("PSA-EVT requires bulk_tau_max == tau_u")
        if int(config["spline_degree"]) < 1:
            raise ValueError(
                "PSA-EVT endpoint curvature requires spline_degree >= 1"
            )

        self.bulk = ISplineBulkQuantile(
            d_model,
            config["num_basis"],
            config["spline_degree"],
            0.5,
            bulk_upper,
            config["coefficient_epsilon"],
        )

        self.tail = PenultimateShapeAdaptiveEVTTail(
            d_model,
            config["tau_u"],
            config["second_order_rho"],
            config["correction_epsilon"],
            config.get("shared_gradient_scale", 1.0),
        )

        self.direct = _ReservedHeadParameters(
            d_model,
            config["num_basis"],
            config["coefficient_epsilon"],
            config["tau_u"],
        )

    def forward(self, horizon_states, taus):
        """
        按配置模式把未来区域状态转换为最终分位数预测。

        参数
        ----
        horizon_states:
            horizon decoder 输出，[B,H,N,R,D]。

        taus:
            请求的分位水平，[Q] 或 [B,Q]。

        返回
        ----
        QuantileHeadOutput:
            其中最终 quantiles 的形状为 [B,H,N,Q]。
        """

        batch = horizon_states.shape[0]

        taus = normalize_taus(
            taus,
            batch,
            horizon_states.device,
            horizon_states.dtype,
            self.tau_max,
        )


        median, coefficients = self.bulk.parameters_from_states(
            horizon_states
        )

        bulk_quantiles, _ = self.bulk.evaluate(
            median,
            coefficients,
            taus,
        )

        threshold, slope, curvature = self.bulk.endpoint_jet(
            median,
            coefficients,
        )


        tail_quantiles, sigma, xi = self.tail(
            threshold,
            slope,
            curvature,
            horizon_states[:, :, :, 2],
            taus,
        )

        quantiles = torch.where(
            (taus > self.tau_u)[:, None, None, :],
            tail_quantiles,
            bulk_quantiles,
        )

        return QuantileHeadOutput(
            quantiles,
            median,
            coefficients,
            threshold,
            slope,
            sigma,
            xi,
        )
