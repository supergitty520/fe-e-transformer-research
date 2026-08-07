"""Finite-element-inspired regularization of hidden-state adjoint gradients.

The trainable path is deliberately explicit: gradients of the task loss with
respect to intermediate hidden states are built with ``create_graph=True``.
Consequently, optimizing these terms is double backpropagation. Calling this
module with ``create_graph=False`` is supported only for detached diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class FETerms:
    """Differentiable scalar terms assembled on a one-dimensional depth mesh."""

    stiffness_raw: Tensor
    stiffness_normalized: Tensor
    mass_energy: Tensor
    node_energy: Tensor
    shannon_entropy: Tensor
    shannon_normalized: Tensor
    relative_entropy: Tensor
    relative_coverage: Tensor
    entropy_score: Tensor


@dataclass
class RegularizerOutput:
    """Objective contribution and detached metrics for logging."""

    penalty: Tensor
    stiffness_penalty: Tensor
    energy_penalty: Tensor
    entropy_penalty: Tensor
    terms: FETerms
    reference_energy: Tensor

    def metrics(self) -> dict[str, float]:
        values = {
            "regularizer": self.penalty,
            "stiffness_penalty": self.stiffness_penalty,
            "energy_penalty": self.energy_penalty,
            "entropy_penalty": self.entropy_penalty,
            "stiffness_raw": self.terms.stiffness_raw,
            "stiffness_normalized": self.terms.stiffness_normalized,
            "mass_energy": self.terms.mass_energy,
            "shannon_entropy": self.terms.shannon_entropy,
            "shannon_normalized": self.terms.shannon_normalized,
            "relative_entropy": self.terms.relative_entropy,
            "relative_coverage": self.terms.relative_coverage,
            "entropy_score": self.terms.entropy_score,
            "reference_energy": self.reference_energy,
        }
        return {name: float(value.detach().cpu()) for name, value in values.items()}


def _validate_gradients(gradients: Sequence[Tensor]) -> None:
    if len(gradients) < 2:
        raise ValueError("At least two depth nodes are required")
    shape = gradients[0].shape
    if not shape:
        raise ValueError("Gradient nodes must be vector- or tensor-valued")
    if any(gradient.shape != shape for gradient in gradients[1:]):
        raise ValueError("All hidden-state gradients must have the same shape")


def _positions_for(
    gradients: Sequence[Tensor], positions: Tensor | Sequence[float] | None
) -> Tensor:
    first = gradients[0]
    if positions is None:
        result = torch.linspace(
            0.0, 1.0, len(gradients), dtype=first.dtype, device=first.device
        )
    else:
        result = torch.as_tensor(positions, dtype=first.dtype, device=first.device)
    if result.ndim != 1 or result.numel() != len(gradients):
        raise ValueError("positions must have one scalar for each depth node")
    if not bool(torch.all(result[1:] > result[:-1])):
        raise ValueError("positions must be strictly increasing")
    return result


def _lumped_weights(positions: Tensor) -> Tensor:
    element_size = positions[1:] - positions[:-1]
    weights = torch.zeros_like(positions)
    weights[:-1] = weights[:-1] + 0.5 * element_size
    weights[1:] = weights[1:] + 0.5 * element_size
    return weights


def _masked_mean(value: Tensor, mask: Tensor | None) -> Tensor:
    if mask is None:
        return value.mean()
    if mask.shape != value.shape[:-1]:
        raise ValueError(
            "mask must match all hidden-state dimensions except the feature dimension"
        )
    weight = mask.to(dtype=value.dtype, device=value.device).unsqueeze(-1)
    denominator = weight.sum() * value.shape[-1]
    return (value * weight).sum() / denominator.clamp_min(1.0)


def _masked_dot(left: Tensor, right: Tensor, mask: Tensor | None) -> Tensor:
    return _masked_mean(left * right, mask)


def fe_entropy_terms(
    gradients: Sequence[Tensor],
    *,
    positions: Tensor | Sequence[float] | None = None,
    mask: Tensor | None = None,
    stiffness_mode: str = "vector",
    entropy_mode: str = "relative_coverage",
    eps: float = 1e-12,
) -> FETerms:
    """Assemble consistent 1-D FE energies and depth-distribution entropy.

    ``vector`` stiffness is the actual H1 seminorm and detects direction
    changes. ``norm`` stiffness is the cheaper conceptual proxy from the brief;
    it sees only changes in RMS gradient magnitude.

    ``relative_coverage`` is ``exp(-KL(p || r))``, where ``r`` is the FE
    quadrature measure. It equals one for a constant energy density even though
    boundary nodes have half weights. ``shannon_normalized`` reproduces the
    normalized Shannon construction more literally.
    """

    _validate_gradients(gradients)
    if stiffness_mode not in {"vector", "norm"}:
        raise ValueError("stiffness_mode must be 'vector' or 'norm'")
    if entropy_mode not in {"relative_coverage", "shannon_normalized"}:
        raise ValueError(
            "entropy_mode must be 'relative_coverage' or 'shannon_normalized'"
        )

    z = _positions_for(gradients, positions)
    element_size = z[1:] - z[:-1]
    weights = _lumped_weights(z)
    node_energy = torch.stack([_masked_mean(g.square(), mask) for g in gradients])

    stiffness_elements: list[Tensor] = []
    mass_elements: list[Tensor] = []
    for index, h in enumerate(element_size):
        left = gradients[index]
        right = gradients[index + 1]
        if stiffness_mode == "vector":
            difference_energy = _masked_mean((right - left).square(), mask)
        else:
            left_rms = torch.sqrt(node_energy[index] + eps)
            right_rms = torch.sqrt(node_energy[index + 1] + eps)
            difference_energy = (right_rms - left_rms).square()
        stiffness_elements.append(difference_energy / h)

        # [a,b] h/6 [[2,1],[1,2]] [a,b]^T, normalized per component.
        cross = _masked_dot(left, right, mask)
        local_mass = h * (node_energy[index] + cross + node_energy[index + 1]) / 3.0
        mass_elements.append(local_mass)

    stiffness_raw = torch.stack(stiffness_elements).sum()
    mass_energy = torch.stack(mass_elements).sum().clamp_min(0.0)
    stiffness_normalized = stiffness_raw / (mass_energy + eps)

    weighted_energy = weights * node_energy
    probability = (weighted_energy + eps * weights) / (
        weighted_energy.sum() + eps * weights.sum()
    )
    reference_probability = weights / weights.sum()
    shannon = -(probability * torch.log(probability.clamp_min(eps))).sum()
    shannon_normalized = shannon / math.log(len(gradients))
    relative_entropy = (
        probability
        * torch.log((probability / reference_probability).clamp_min(eps))
    ).sum()
    relative_coverage = torch.exp(-relative_entropy)
    entropy_score = (
        relative_coverage
        if entropy_mode == "relative_coverage"
        else shannon_normalized
    )

    return FETerms(
        stiffness_raw=stiffness_raw,
        stiffness_normalized=stiffness_normalized,
        mass_energy=mass_energy,
        node_energy=node_energy,
        shannon_entropy=shannon,
        shannon_normalized=shannon_normalized,
        relative_entropy=relative_entropy,
        relative_coverage=relative_coverage,
        entropy_score=entropy_score,
    )


class FEERegularizer(nn.Module):
    """FE-E double-backpropagation regularizer for residual-state sequences."""

    def __init__(
        self,
        *,
        lambda_stiffness: float = 0.0,
        lambda_energy: float = 0.0,
        lambda_entropy: float = 0.0,
        entropy_band: tuple[float, float] = (0.65, 0.98),
        stiffness_mode: str = "vector",
        entropy_mode: str = "relative_coverage",
        energy_penalty: str = "log_ratio",
        reference_mode: str = "fixed_initial",
        ema_decay: float = 0.99,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        if not 0.0 <= entropy_band[0] <= entropy_band[1] <= 1.0:
            raise ValueError("entropy_band must lie in [0, 1]")
        if energy_penalty not in {"log_ratio", "ratio"}:
            raise ValueError("energy_penalty must be 'log_ratio' or 'ratio'")
        if reference_mode not in {"fixed_initial", "ema"}:
            raise ValueError("reference_mode must be 'fixed_initial' or 'ema'")
        if not 0.0 <= ema_decay < 1.0:
            raise ValueError("ema_decay must lie in [0, 1)")

        self.lambda_stiffness = float(lambda_stiffness)
        self.lambda_energy = float(lambda_energy)
        self.lambda_entropy = float(lambda_entropy)
        self.entropy_band = entropy_band
        self.stiffness_mode = stiffness_mode
        self.entropy_mode = entropy_mode
        self.energy_penalty_kind = energy_penalty
        self.reference_mode = reference_mode
        self.ema_decay = ema_decay
        self.eps = eps
        self.register_buffer("energy_reference", torch.tensor(float("nan")))

    @property
    def is_active(self) -> bool:
        return any(
            coefficient != 0.0
            for coefficient in (
                self.lambda_stiffness,
                self.lambda_energy,
                self.lambda_entropy,
            )
        )

    def reset_reference(self) -> None:
        self.energy_reference.fill_(float("nan"))

    def _reference_for(self, energy: Tensor, external: Tensor | float | None) -> Tensor:
        if external is not None:
            return torch.as_tensor(external, dtype=energy.dtype, device=energy.device)
        if bool(torch.isnan(self.energy_reference)):
            self.energy_reference.copy_(energy.detach().to(self.energy_reference))
        reference = self.energy_reference.to(dtype=energy.dtype, device=energy.device)
        if self.training and self.reference_mode == "ema":
            updated = self.ema_decay * reference + (1.0 - self.ema_decay) * energy.detach()
            self.energy_reference.copy_(updated.to(self.energy_reference))
        return reference.detach()

    def forward(
        self,
        task_loss: Tensor,
        hidden_states: Sequence[Tensor],
        *,
        positions: Tensor | Sequence[float] | None = None,
        mask: Tensor | None = None,
        create_graph: bool = True,
        reference_energy: Tensor | float | None = None,
    ) -> RegularizerOutput:
        if self.is_active and not create_graph:
            raise ValueError(
                "An active gradient regularizer requires create_graph=True; "
                "detached hooks can only monitor it"
            )
        gradients = torch.autograd.grad(
            task_loss,
            tuple(hidden_states),
            create_graph=create_graph,
            retain_graph=True,
            allow_unused=False,
        )
        terms = fe_entropy_terms(
            gradients,
            positions=positions,
            mask=mask,
            stiffness_mode=self.stiffness_mode,
            entropy_mode=self.entropy_mode,
            eps=self.eps,
        )
        reference = self._reference_for(terms.mass_energy, reference_energy)
        if self.energy_penalty_kind == "log_ratio":
            energy_penalty = torch.log(
                (terms.mass_energy + self.eps) / (reference + self.eps)
            ).square()
        else:
            energy_penalty = (
                terms.mass_energy / (reference + self.eps) - 1.0
            ).square()

        lower, upper = self.entropy_band
        entropy_penalty = F.relu(lower - terms.entropy_score).square() + F.relu(
            terms.entropy_score - upper
        ).square()
        stiffness_penalty = terms.stiffness_normalized
        penalty = (
            self.lambda_stiffness * stiffness_penalty
            + self.lambda_energy * energy_penalty
            + self.lambda_entropy * entropy_penalty
        )
        return RegularizerOutput(
            penalty=penalty,
            stiffness_penalty=stiffness_penalty,
            energy_penalty=energy_penalty,
            entropy_penalty=entropy_penalty,
            terms=terms,
            reference_energy=reference,
        )


# Backward-compatible public name used by the first research prototype.
FEEntropyRegularizer = FEERegularizer
