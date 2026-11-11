import torch
import torch.nn as nn


def _bspline_basis(x, knots, degree):
    x = x.unsqueeze(-1)
    right_limit = torch.nextafter(knots[-1], knots[0])
    x = torch.where(x == knots[-1], right_limit, x)
    basis = ((x >= knots[:-1]) & (x < knots[1:])).to(dtype=x.dtype)
    for order in range(1, degree + 1):
        count = knots.numel() - order - 1
        left_denominator = knots[order : order + count] - knots[:count]
        right_denominator = knots[order + 1 : order + count + 1] - knots[1 : count + 1]
        left_weight = torch.zeros_like(x)
        right_weight = torch.zeros_like(x)
        valid_left = left_denominator > 0
        valid_right = right_denominator > 0
        left_weight = torch.where(
            valid_left, (x - knots[:count]) / left_denominator.clamp_min(1e-30), left_weight
        )
        right_weight = torch.where(
            valid_right,
            (knots[order + 1 : order + count + 1] - x)
            / right_denominator.clamp_min(1e-30),
            right_weight,
        )
        basis = left_weight * basis[..., :count] + right_weight * basis[..., 1 : count + 1]
    return basis


def _bspline_derivative_basis(x, knots, degree):
    if degree < 1:
        raise ValueError("B-spline derivative requires degree >= 1")
    lower_basis = _bspline_basis(x, knots, degree - 1)
    count = knots.numel() - degree - 1
    left_denominator = knots[degree : degree + count] - knots[:count]
    right_denominator = (
        knots[degree + 1 : degree + count + 1] - knots[1 : count + 1]
    )
    left_scale = torch.where(
        left_denominator > 0,
        degree / left_denominator.clamp_min(1e-30),
        torch.zeros_like(left_denominator),
    )
    right_scale = torch.where(
        right_denominator > 0,
        degree / right_denominator.clamp_min(1e-30),
        torch.zeros_like(right_denominator),
    )
    return (
        left_scale * lower_basis[..., :count]
        - right_scale * lower_basis[..., 1 : count + 1]
    )


class ISplineBasis(nn.Module):
    def __init__(self, num_basis, degree, lower=0.5, upper=0.9):
        super().__init__()
        if degree < 0 or num_basis <= degree:
            raise ValueError("num_basis must be greater than the non-negative spline degree")
        if not lower < upper:
            raise ValueError("I-spline lower endpoint must be smaller than upper endpoint")
        interior_count = num_basis - degree - 1
        interior = torch.linspace(lower, upper, interior_count + 2, dtype=torch.float64)[1:-1]
        knots = torch.cat(
            (
                torch.full((degree + 1,), lower, dtype=torch.float64),
                interior,
                torch.full((degree + 1,), upper, dtype=torch.float64),
            )
        )
        antiderivative_knots = torch.cat((knots[:1], knots, knots[-1:]))
        coefficients = torch.zeros(num_basis, num_basis + 1, dtype=torch.float64)
        for index in range(num_basis):
            coefficients[index, index + 1 :] = 1.0
        centers = 0.5 * (knots[:num_basis] + knots[degree + 1 : degree + 1 + num_basis])
        self.num_basis = int(num_basis)
        self.degree = int(degree)
        self.lower = float(lower)
        self.upper = float(upper)
        self.register_buffer("knots", knots)
        self.register_buffer("antiderivative_knots", antiderivative_knots)
        self.register_buffer("antiderivative_coefficients", coefficients)
        self.register_buffer("support_centers", centers)

    def forward(self, tau):
        if torch.any(tau < self.lower) or torch.any(tau > self.upper):
            raise ValueError(f"I-spline tau must lie in [{self.lower}, {self.upper}]")
        knots = self.knots.to(device=tau.device, dtype=tau.dtype)
        spans = knots[self.degree + 1 :] - knots[: -(self.degree + 1)]
        b_basis = _bspline_basis(tau, knots, self.degree)
        m_basis = (self.degree + 1) * b_basis / spans
        anti_knots = self.antiderivative_knots.to(device=tau.device, dtype=tau.dtype)
        anti_basis = _bspline_basis(tau, anti_knots, self.degree + 1)
        coefficients = self.antiderivative_coefficients.to(device=tau.device, dtype=tau.dtype)
        i_basis = torch.einsum("...k,jk->...j", anti_basis, coefficients)
        return i_basis, m_basis

    def endpoint_left(self, dtype=None, device=None):
        dtype = dtype or self.knots.dtype
        device = device or self.knots.device
        tau = torch.tensor(self.upper, dtype=dtype, device=device)
        return self(tau)

    def endpoint_left_jet(self, dtype=None, device=None):
        if self.degree < 1:
            raise ValueError("endpoint curvature requires spline degree >= 1")
        dtype = dtype or self.knots.dtype
        device = device or self.knots.device
        tau = torch.tensor(self.upper, dtype=dtype, device=device)
        i_basis, m_basis = self(tau)
        knots = self.knots.to(device=device, dtype=dtype)
        spans = knots[self.degree + 1 :] - knots[: -(self.degree + 1)]
        db_basis = _bspline_derivative_basis(tau, knots, self.degree)
        dm_basis = (self.degree + 1) * db_basis / spans
        return i_basis, m_basis, dm_basis


class ISplineBulkQuantile(nn.Module):
    def __init__(
        self,
        d_model,
        num_basis,
        degree,
        lower,
        upper,
        coefficient_epsilon,
    ):
        super().__init__()
        self.basis = ISplineBasis(num_basis, degree, lower, upper)
        self.coefficient_epsilon = float(coefficient_epsilon)
        self.median_proj = nn.Linear(d_model, 1)
        self.median_coefficients = nn.Linear(d_model, num_basis)
        self.upper_coefficients = nn.Linear(d_model, num_basis)
        midpoint = 0.5 * (lower + upper)
        assignment = (self.basis.support_centers > midpoint).to(torch.bool)
        self.register_buffer("upper_assignment", assignment)

    def parameters_from_states(self, region_states):
        if region_states.ndim != 5 or region_states.shape[3] != 3:
            raise ValueError("bulk head expects horizon states [B,H,N,3,D]")
        median_state = region_states[:, :, :, 0]
        upper_state = region_states[:, :, :, 1]
        median = self.median_proj(median_state)
        raw_median = self.median_coefficients(median_state)
        raw_upper = self.upper_coefficients(upper_state)
        raw = torch.where(self.upper_assignment, raw_upper, raw_median)
        coefficients = self.coefficient_epsilon + torch.nn.functional.softplus(raw)
        return median, coefficients

    def evaluate(self, median, coefficients, taus):
        clipped = taus.clamp(min=self.basis.lower, max=self.basis.upper)
        i_basis, m_basis = self.basis(clipped)
        lower_tau = clipped.new_tensor(self.basis.lower)
        lower_i, _ = self.basis(lower_tau)
        centered_i = i_basis - lower_i
        quantiles = median + torch.einsum("bhnj,bqj->bhnq", coefficients, centered_i)
        derivative = torch.einsum("bhnj,bqj->bhnq", coefficients, m_basis)
        return quantiles, derivative

    def endpoint(self, median, coefficients):
        i_basis, m_basis = self.basis.endpoint_left(
            dtype=median.dtype, device=median.device
        )
        lower_i, _ = self.basis(median.new_tensor(self.basis.lower))
        increment = torch.einsum("bhnj,j->bhn", coefficients, i_basis - lower_i)
        slope = torch.einsum("bhnj,j->bhn", coefficients, m_basis).unsqueeze(-1)
        return median + increment.unsqueeze(-1), slope

    def endpoint_jet(self, median, coefficients):
        i_basis, m_basis, dm_basis = self.basis.endpoint_left_jet(
            dtype=median.dtype, device=median.device
        )
        lower_i, _ = self.basis(median.new_tensor(self.basis.lower))
        increment = torch.einsum("bhnj,j->bhn", coefficients, i_basis - lower_i)
        slope = torch.einsum("bhnj,j->bhn", coefficients, m_basis).unsqueeze(-1)
        curvature = torch.einsum(
            "bhnj,j->bhn", coefficients, dm_basis
        ).unsqueeze(-1)
        return median + increment.unsqueeze(-1), slope, curvature
