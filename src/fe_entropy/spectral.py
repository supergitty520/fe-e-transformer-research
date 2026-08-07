"""Small-matrix Jacobian spectral diagnostics.

These functions are exact and intended for audits or randomized low-rank
surrogates, not for materializing full Transformer-block Jacobians.
"""

from __future__ import annotations

import torch
from torch import Tensor


def spectral_metrics(jacobian: Tensor, eps: float = 1e-12) -> dict[str, Tensor]:
    """Return scale-sensitive and scale-free singular-spectrum diagnostics."""

    if jacobian.ndim != 2:
        raise ValueError("jacobian must be a matrix")
    singular_values = torch.linalg.svdvals(jacobian)
    energy = singular_values.square()
    probability = (energy + eps) / (energy.sum() + eps * energy.numel())
    entropy = -(probability * torch.log(probability.clamp_min(eps))).sum()
    entropy_normalized = entropy / torch.log(
        torch.as_tensor(
            max(2, energy.numel()), dtype=energy.dtype, device=energy.device
        )
    )
    rms = torch.sqrt(energy.mean())
    log_isometry_error = torch.log(singular_values.clamp_min(eps)).square().mean()
    stable_rank = energy.sum() / energy.max().clamp_min(eps)
    condition_number = singular_values.max() / singular_values.min().clamp_min(eps)
    return {
        "spectral_entropy": entropy,
        "spectral_entropy_normalized": entropy_normalized,
        "rms_singular_value": rms,
        "log_isometry_error": log_isometry_error,
        "stable_rank": stable_rank,
        "condition_number": condition_number,
    }

