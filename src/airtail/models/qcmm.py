import torch.nn as nn
import torch.nn.functional as F

from airtail.models.selective_ssm import QuantileConditionedMultiscaleSSM


class QuantileConditionedMultiscaleMambaBlock(nn.Module):
    """Mamba shell around the quantile-conditioned multiscale selective SSM."""

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
        d_conv=4,
        expand=2,
        bias=False,
        conv_bias=True,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.num_scales = int(num_scales)
        self.d_conv = int(d_conv)
        self.d_inner = int(expand * d_model)

        if self.d_conv < 1:
            raise ValueError("d_conv must be positive")
        if self.d_inner < 1:
            raise ValueError("expanded channel dimension must be positive")

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=bias)
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=self.d_conv,
            groups=self.d_inner,
            padding=0,
            bias=conv_bias,
        )
        self.act = nn.SiLU()
        self.selective_ssm = QuantileConditionedMultiscaleSSM(
            self.d_inner,
            d_state,
            num_scales,
            delta_min,
            delta_max,
            delta_mode=delta_mode,
            history_gamma=history_gamma,
            backend=backend,
        )
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

    def _causal_conv(self, x):
        """Apply independent depthwise causal convolutions over history."""
        batch, length, stations, scales, channels = x.shape
        x = x.permute(0, 2, 3, 4, 1).reshape(
            batch * stations * scales, channels, length
        )
        x = self.conv1d(F.pad(x, (self.d_conv - 1, 0)))
        x = self.act(x)
        return x.reshape(batch, stations, scales, channels, length).permute(
            0, 4, 1, 2, 3
        )

    def forward(self, x, history_mask=None, return_diagnostics=False):
        if x.ndim != 5:
            raise ValueError(f"quantile Mamba expects [B,L,N,S,D], got {tuple(x.shape)}")
        if x.shape[3] != self.num_scales or x.shape[4] != self.d_model:
            raise ValueError(
                "quantile Mamba scale or channel dimension does not match its configuration"
            )

        x_branch, z_branch = self.in_proj(x).chunk(2, dim=-1)
        x_branch = self._causal_conv(x_branch)
        y, delta_summary, half_life_summary, diagnostics = self.selective_ssm(
            x_branch,
            history_mask,
            return_diagnostics=return_diagnostics,
        )
        y = y * F.silu(z_branch).unsqueeze(3)
        y = self.out_proj(y)
        return y, delta_summary, half_life_summary, diagnostics
