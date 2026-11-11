"""Metrics defined in the manuscript, accumulated in concentration units."""

import numpy as np
import torch


class PaperMetrics:
    def __init__(self, taus, scaler):
        self.taus = tuple(taus)
        self.lower = self.taus.index(0.5)
        self.upper = self.taus.index(0.99)
        self.mean = float(np.asarray(scaler.mean).reshape(-1)[0])
        self.std = float(np.asarray(scaler.std).reshape(-1)[0])
        self.sums = None

    def update(self, prediction, target, mask):
        valid = mask.detach().cpu()[..., 0]
        p = prediction.detach().double().cpu() * self.std + self.mean
        y = target.detach().double().cpu() * self.std + self.mean
        taus = torch.tensor(self.taus, dtype=torch.float64)
        error = y - p
        pinball = torch.maximum(taus * error, (taus - 1.0) * error)
        low = error[..., self.lower]
        high = error[..., self.upper]
        interval = (low >= 0) & (high <= 0)
        score = (p[..., self.upper] - p[..., self.lower]
                 + (-low).clamp_min(0) / 0.5 + high.clamp_min(0) / 0.01)
        stats = torch.cat((
            pinball, y.abs(), interval.unsqueeze(-1), score.unsqueeze(-1),
            torch.ones_like(y),
        ), dim=-1)
        totals = torch.where(valid.unsqueeze(-1), stats, 0.0).sum(dim=(0, 2))
        self.sums = totals if self.sums is None else self.sums + totals

    def _compute(self, sums):
        pinball = sums[:len(self.taus)]
        target, covered, interval_score, count = sums[len(self.taus):].tolist()
        risk = (2 * pinball / target).tolist() if target else [None] * len(self.taus)
        return {
            "q_risk_by_tau": dict(zip(map(str, self.taus), risk)),
            "mean_q_risk": sum(risk) / len(risk) if target else None,
            "picp": covered / count if count else None,
            "qis": interval_score / count if count else None,
            "valid_count": int(count),
        }

    def compute(self):
        if self.sums is None:
            raise ValueError("no evaluation batches were accumulated")
        return {
            **self._compute(self.sums.sum(dim=0)),
            "by_horizon_step": {
                str(i): self._compute(row) for i, row in enumerate(self.sums, start=1)
            },
        }
