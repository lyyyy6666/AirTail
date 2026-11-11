import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _exprel(x):
    result = torch.empty_like(x)
    small = x.abs() < 1e-4
    if small.any():
        value = x[small]
        result[small] = (
            1.0
            + value / 2.0
            + value.square() / 6.0
            + value.pow(3) / 24.0
        )
    if (~small).any():
        value = x[~small]
        result[~small] = torch.expm1(value) / value
    return result


def _pareto_increment(alpha, log_r):
    return log_r * _exprel(alpha * log_r)


class PenultimateShapeAdaptiveEVTTail(nn.Module):
    def __init__(
        self,
        d_model,
        tau_u,
        rho=-1.0,
        correction_epsilon=1e-4,
        shared_gradient_scale=1.0,
    ):
        super().__init__()
        if not 0.5 < tau_u < 1.0:
            raise ValueError("tau_u must lie strictly between 0.5 and 1")
        if rho >= 0.0:
            raise ValueError("second-order rho must be negative")
        if not 0.0 < correction_epsilon < -rho:
            raise ValueError("correction_epsilon must lie in (0, -rho)")
        if not 0.0 <= shared_gradient_scale <= 1.0:
            raise ValueError("shared_gradient_scale must lie in [0, 1]")
        self.tau_u = float(tau_u)
        self.rho = float(rho)
        self.correction_epsilon = float(correction_epsilon)
        self.shared_gradient_scale = float(shared_gradient_scale)
        self.correction_proj = nn.Linear(d_model, 1)
        nn.init.zeros_(self.correction_proj.weight)
        target = -self.rho - self.correction_epsilon
        nn.init.constant_(self.correction_proj.bias, math.log(math.expm1(target)))

    def tail_parameters(self, slope, curvature, extreme_state):
        sigma = (1.0 - self.tau_u) * slope
        kappa = -1.0 + (1.0 - self.tau_u) * curvature / slope
        correction = (
            self.rho
            + self.correction_epsilon
            + F.softplus(self.correction_proj(extreme_state))
        )
        xi = kappa - correction
        self._check_finite(
            sigma=sigma,
            kappa=kappa,
            correction=correction,
            xi=xi,
        )
        return sigma, kappa, correction, xi

    def second_order_basis(self, xi, log_r):
        first = _pareto_increment(xi, log_r)
        shifted = _pareto_increment(xi + self.rho, log_r)
        return first, (shifted - first) / self.rho

    def evaluate(self, threshold, slope, correction, xi, taus):
        sigma = (1.0 - self.tau_u) * slope
        tail_tau = taus.clamp(min=self.tau_u)
        log_r = (
            tail_tau.new_tensor(math.log1p(-self.tau_u))
            - torch.log1p(-tail_tau)
        )[:, None, None, :]
        first, second = self.second_order_basis(xi, log_r)
        monotone_factor = 1.0 + correction * _pareto_increment(
            log_r.new_tensor(self.rho), log_r
        )
        quantiles = threshold + sigma * (first + correction * second)
        self._check_finite(
            log_r=log_r,
            first_order=first,
            second_order=second,
            monotone_factor=monotone_factor,
            quantiles=quantiles,
        )
        if torch.any(monotone_factor <= 0):
            raise FloatingPointError("PSA-EVT monotonicity factor is non-positive")
        return quantiles, sigma

    def forward(self, threshold, slope, curvature, extreme_state, taus):
        threshold = self._scale_shared_gradient(threshold)
        slope = self._scale_shared_gradient(slope)
        curvature = self._scale_shared_gradient(curvature)
        extreme_state = self._scale_shared_gradient(extreme_state)
        sigma, _, correction, xi = self.tail_parameters(
            slope, curvature, extreme_state
        )
        quantiles, _ = self.evaluate(
            threshold, slope, correction, xi, taus
        )
        return quantiles, sigma, xi

    def _scale_shared_gradient(self, value):
        scale = self.shared_gradient_scale
        return value.detach() + scale * (value - value.detach())

    @staticmethod
    def _check_finite(**values):
        invalid = [
            name
            for name, value in values.items()
            if not torch.isfinite(value).all()
        ]
        if invalid:
            raise FloatingPointError(
                f"PSA-EVT produced non-finite values in {', '.join(invalid)}"
            )
