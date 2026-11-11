import torch


def weighted_integrated_pinball_parts(
    prediction, target, target_mask, taus, tau_weights=None
):
    if prediction.ndim != 4:
        raise ValueError("prediction must have shape [B,H,N,Q]")
    if target.shape != prediction.shape[:3] + (1,):
        raise ValueError("target must have shape [B,H,N,1]")
    if target_mask.ndim == 3:
        target_mask = target_mask.unsqueeze(-1)
    if target_mask.shape[:3] != target.shape[:3] or target_mask.ndim != 4:
        raise ValueError("target_mask must have shape [B,H,N] or [B,H,N,1]")
    batch, _, _, quantile_count = prediction.shape
    if taus.ndim == 1:
        taus = taus[None].expand(batch, -1)
    if taus.shape != (batch, quantile_count):
        raise ValueError("taus must match prediction batch and quantile dimensions")
    taus = taus.to(device=prediction.device, dtype=prediction.dtype)
    valid = target_mask.to(dtype=prediction.dtype).any(dim=-1, keepdim=True)
    clean_target = torch.where(valid, target, torch.zeros_like(target))
    clean_prediction = torch.where(
        valid.expand_as(prediction), prediction, torch.zeros_like(prediction)
    )
    error = clean_target - clean_prediction
    loss = torch.maximum(
        taus[:, None, None] * error,
        (taus[:, None, None] - 1.0) * error,
    )
    if tau_weights is None:
        weights = torch.ones_like(taus)
    else:
        weights = tau_weights.to(device=prediction.device, dtype=prediction.dtype)
        if weights.ndim == 1:
            weights = weights[None].expand(batch, -1)
        if weights.shape != taus.shape or torch.any(weights < 0):
            raise ValueError("tau_weights must be non-negative and match taus")
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    weighted = loss * weights[:, None, None] * valid
    numerator = weighted.sum()
    denominator = valid.sum()
    return numerator, denominator, numerator / denominator.clamp_min(1.0)


def weighted_integrated_pinball(prediction, target, target_mask, taus, tau_weights=None):
    return weighted_integrated_pinball_parts(
        prediction, target, target_mask, taus, tau_weights
    )[2]
