"""Apple MLX experiment for always-on, periodic, and observer-gated FE-E.

This module is intentionally self-contained.  It uses a manual reverse VJP
recursion to expose task-loss adjoints at every residual-stream node.  Nesting
that recursion inside ``nn.value_and_grad`` gives exact double
backpropagation on MLX without materializing full Jacobians.

Every training step is synchronously evaluated and appended to JSONL before
the next step starts.  The logs are therefore suitable for failure recovery
and paper-audit use, not only for console monitoring.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import sys
import time
from typing import Any, Iterable, Sequence

import mlx
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map, tree_unflatten


VARIANTS = (
    "baseline",
    "gradient_smoothing",
    "fe_e_always",
    # Retained only to reproduce the negative pulse-ablation experiments.
    "fe_e_periodic",
    "fe_e_gated",
    "gs_fe_e_gated",
    "adamw_observer_control",
    "gs_observer_control",
)

DEFAULT_VARIANTS = (
    "baseline",
    "gradient_smoothing",
    "fe_e_always",
    "fe_e_gated",
    "gs_fe_e_gated",
)


@dataclass(frozen=True)
class ExperimentConfig:
    steps: int = 200
    layers: int = 24
    width: int = 32
    heads: int = 4
    sequence_length: int = 12
    batch_size: int = 8
    vocab_size: int = 32
    learning_rate: float = 0.002
    weight_decay: float = 0.01
    residual_scale: float = 1.0
    lambda_stiffness: float = 2.0
    lambda_energy: float = 0.02
    lambda_entropy: float = 2.0
    entropy_lower: float = 0.90
    entropy_upper: float = 0.98
    regularizer_warmup_steps: int = 10
    periodic_every: int = 8
    diagnostic_every: int = 20
    observer_probe_every: int = 8
    evaluation_every: int = 25
    evaluation_batches: int = 10
    smoothing_alpha: float = 0.20
    observer_calibration_steps: int = 24
    observer_on_threshold: float = 2.5
    observer_off_threshold: float = 1.0
    observer_persistence_window: int = 4
    observer_persistence_required: int = 3
    observer_metric_votes_required: int = 2
    observer_harm_consecutive: int = 2
    observer_sentinel_loss_tolerance: float = 0.01
    observer_sentinel_accuracy_tolerance: float = 0.01
    observer_phase_improvement_tolerance: float = 0.02
    observer_baseline_window: int = 16
    observer_adaptive_baseline: bool = False
    intervention_steps: int = 1
    recovery_steps: int = 48
    observer_mass_log_tolerance: float = 0.50
    observer_coverage_floor: float = 0.90
    log_fsync_every: int = 20
    timing_warmup_steps: int = 5
    stress_step: int = -1
    stress_duration: int = 8
    stress_lr_multiplier: float = 3.0
    gated_fee_gradient_ratio: float = 0.05
    gated_fee_gradient_ratio_max: float = 0.20
    target_token_accuracy: float = 0.99
    target_confirmations: int = 3
    stop_on_target: bool = False

    def validate(self) -> None:
        positive = (
            "steps",
            "layers",
            "width",
            "heads",
            "sequence_length",
            "batch_size",
            "vocab_size",
            "periodic_every",
            "diagnostic_every",
            "observer_probe_every",
            "evaluation_every",
            "evaluation_batches",
            "observer_calibration_steps",
            "observer_persistence_window",
            "observer_persistence_required",
            "observer_metric_votes_required",
            "observer_harm_consecutive",
            "observer_baseline_window",
            "target_confirmations",
            "intervention_steps",
            "recovery_steps",
            "log_fsync_every",
            "timing_warmup_steps",
            "stress_duration",
        )
        for name in positive:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.width % self.heads:
            raise ValueError("width must be divisible by heads")
        if not 0.0 <= self.smoothing_alpha < 1.0:
            raise ValueError("smoothing_alpha must lie in [0, 1)")
        if not 0.0 <= self.entropy_lower <= self.entropy_upper <= 1.0:
            raise ValueError("entropy band must lie in [0, 1]")
        if self.observer_off_threshold >= self.observer_on_threshold:
            raise ValueError("observer_off_threshold must be below on_threshold")
        if self.observer_persistence_required > self.observer_persistence_window:
            raise ValueError(
                "observer_persistence_required cannot exceed persistence_window"
            )
        if self.observer_metric_votes_required > 4:
            raise ValueError("observer_metric_votes_required cannot exceed 4")
        for name in (
            "observer_sentinel_loss_tolerance",
            "observer_sentinel_accuracy_tolerance",
            "observer_phase_improvement_tolerance",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.stress_step < -1:
            raise ValueError("stress_step must be -1 (disabled) or non-negative")
        if self.stress_lr_multiplier <= 0.0:
            raise ValueError("stress_lr_multiplier must be positive")
        if self.gated_fee_gradient_ratio <= 0.0:
            raise ValueError("gated_fee_gradient_ratio must be positive")
        if self.gated_fee_gradient_ratio_max < self.gated_fee_gradient_ratio:
            raise ValueError(
                "gated_fee_gradient_ratio_max must not be below the base ratio"
            )
        if not 0.0 < self.target_token_accuracy <= 1.0:
            raise ValueError("target_token_accuracy must lie in (0, 1]")


class SelfAttention(nn.Module):
    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.head_width = width // heads
        self.qkv = nn.Linear(width, 3 * width)
        self.output = nn.Linear(width, width)

    def __call__(self, hidden: mx.array) -> mx.array:
        batch, length, width = hidden.shape
        qkv = self.qkv(hidden).reshape(
            batch, length, 3, self.heads, self.head_width
        )
        query, key, value = [qkv[:, :, index] for index in range(3)]
        query = query.transpose(0, 2, 1, 3)
        key = key.transpose(0, 2, 1, 3)
        value = value.transpose(0, 2, 1, 3)
        scores = (query @ key.transpose(0, 1, 3, 2)) / math.sqrt(self.head_width)
        attention = mx.softmax(scores, axis=-1)
        mixed = attention @ value
        mixed = mixed.transpose(0, 2, 1, 3).reshape(batch, length, width)
        return self.output(mixed)


class FeedForward(nn.Module):
    def __init__(self, width: int, expansion: int = 2) -> None:
        super().__init__()
        self.up = nn.Linear(width, expansion * width)
        self.down = nn.Linear(expansion * width, width)

    def __call__(self, hidden: mx.array) -> mx.array:
        return self.down(nn.gelu(self.up(hidden)))


class TransformerBlock(nn.Module):
    def __init__(self, width: int, heads: int, residual_scale: float) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(width)
        self.attention = SelfAttention(width, heads)
        self.mlp_norm = nn.LayerNorm(width)
        self.mlp = FeedForward(width)
        self.residual_scale = residual_scale

    def __call__(self, hidden: mx.array) -> mx.array:
        hidden = hidden + self.residual_scale * self.attention(
            self.attention_norm(hidden)
        )
        hidden = hidden + self.residual_scale * self.mlp(self.mlp_norm(hidden))
        return hidden


class TinyTransformer(nn.Module):
    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.width)
        self.position_embedding = mx.random.normal(
            (config.sequence_length, config.width)
        ) * 0.02
        self.blocks = [
            TransformerBlock(config.width, config.heads, config.residual_scale)
            for _ in range(config.layers)
        ]
        self.final_norm = nn.LayerNorm(config.width)
        self.readout = nn.Linear(config.width, config.vocab_size, bias=False)

    def forward_states(self, tokens: mx.array) -> list[mx.array]:
        hidden = self.token_embedding(tokens) + self.position_embedding[: tokens.shape[1]]
        states = [hidden]
        for block in self.blocks:
            hidden = block(hidden)
            states.append(hidden)
        return states

    def logits_from_hidden(self, hidden: mx.array) -> mx.array:
        return self.readout(self.final_norm(hidden))

    def __call__(self, tokens: mx.array) -> mx.array:
        return self.logits_from_hidden(self.forward_states(tokens)[-1])


def reverse_sequence_loss(logits: mx.array, tokens: mx.array) -> mx.array:
    targets = tokens[:, ::-1]
    return nn.losses.cross_entropy(logits, targets, reduction="mean")


def hidden_adjoint_field(
    model: TinyTransformer,
    states: Sequence[mx.array],
    tokens: mx.array,
) -> list[mx.array]:
    """Return d(task loss)/d(h_l) for every residual node in O(L) VJPs."""

    def head_loss(hidden: mx.array) -> mx.array:
        return reverse_sequence_loss(model.logits_from_hidden(hidden), tokens)

    gradient = mx.grad(head_loss)(states[-1])
    reverse_gradients = [gradient]
    for index in range(len(model.blocks) - 1, -1, -1):
        _, vjps = mx.vjp(model.blocks[index], [states[index]], [gradient])
        gradient = vjps[0]
        reverse_gradients.append(gradient)
    return list(reversed(reverse_gradients))


def fe_terms(gradients: Sequence[mx.array], eps: float = 1e-12) -> dict[str, mx.array]:
    """Finite-element terms on a unit-spaced layer-index mesh."""

    if len(gradients) < 2:
        raise ValueError("at least two hidden adjoints are required")
    node_energy = mx.stack([mx.mean(mx.square(g)) for g in gradients])
    stiffness_elements = [
        mx.mean(mx.square(right - left))
        for left, right in zip(gradients, gradients[1:])
    ]
    mass_elements = [
        (node_energy[index] + mx.mean(left * right) + node_energy[index + 1])
        / 3.0
        for index, (left, right) in enumerate(zip(gradients, gradients[1:]))
    ]
    stiffness_raw = mx.sum(mx.stack(stiffness_elements))
    mass_energy = mx.maximum(mx.sum(mx.stack(mass_elements)), mx.array(0.0))
    stiffness_normalized = stiffness_raw / (mass_energy + eps)
    weights = mx.concatenate(
        [mx.array([0.5]), mx.ones((len(gradients) - 2,)), mx.array([0.5])]
    )
    weighted_energy = weights * node_energy
    probability = (weighted_energy + eps * weights) / (
        mx.sum(weighted_energy) + eps * mx.sum(weights)
    )
    reference_probability = weights / mx.sum(weights)
    relative_entropy = mx.sum(
        probability * mx.log(mx.maximum(probability / reference_probability, eps))
    )
    coverage = mx.exp(-relative_entropy)
    shannon = -mx.sum(probability * mx.log(mx.maximum(probability, eps)))
    return {
        "stiffness_raw": stiffness_raw,
        "stiffness_normalized": stiffness_normalized,
        "mass_energy": mass_energy,
        "relative_entropy": relative_entropy,
        "relative_coverage": coverage,
        "shannon_entropy": shannon,
    }


def fee_penalty(
    terms: dict[str, mx.array],
    reference_energy: float | None,
    config: ExperimentConfig,
) -> tuple[mx.array, dict[str, mx.array]]:
    reference = (
        mx.stop_gradient(terms["mass_energy"])
        if reference_energy is None
        else mx.array(reference_energy)
    )
    energy_penalty = mx.square(
        mx.log((terms["mass_energy"] + 1e-12) / (reference + 1e-12))
    )
    coverage = terms["relative_coverage"]
    entropy_penalty = mx.square(mx.maximum(config.entropy_lower - coverage, 0.0))
    entropy_penalty = entropy_penalty + mx.square(
        mx.maximum(coverage - config.entropy_upper, 0.0)
    )
    stiffness_penalty = terms["stiffness_normalized"]
    penalty = (
        config.lambda_stiffness * stiffness_penalty
        + config.lambda_energy * energy_penalty
        + config.lambda_entropy * entropy_penalty
    )
    return penalty, {
        "stiffness_penalty": stiffness_penalty,
        "energy_penalty": energy_penalty,
        "entropy_penalty": entropy_penalty,
        "reference_energy": reference,
    }


def _zeros_terms() -> dict[str, mx.array]:
    zero = mx.array(0.0)
    return {
        "stiffness_raw": zero,
        "stiffness_normalized": zero,
        "mass_energy": zero,
        "relative_entropy": zero,
        "relative_coverage": zero,
        "shannon_entropy": zero,
        "stiffness_penalty": zero,
        "energy_penalty": zero,
        "entropy_penalty": zero,
        "reference_energy": zero,
    }


def loss_with_aux(
    model: TinyTransformer,
    tokens: mx.array,
    compute_adjoint: bool,
    apply_fee: bool,
    penalty_scale: float,
    reference_energy: float | None,
    config: ExperimentConfig,
) -> tuple[mx.array, dict[str, mx.array]]:
    states = model.forward_states(tokens)
    logits = model.logits_from_hidden(states[-1])
    task_loss = reverse_sequence_loss(logits, tokens)
    if compute_adjoint:
        terms = fe_terms(hidden_adjoint_field(model, states, tokens))
        penalty, penalty_terms = fee_penalty(terms, reference_energy, config)
        terms = {**terms, **penalty_terms}
    else:
        terms = _zeros_terms()
        penalty = mx.array(0.0)
    total_loss = task_loss + penalty_scale * penalty if apply_fee else task_loss
    increments = mx.stack(
        [right - left for left, right in zip(states, states[1:])]
    )
    increment_norms = mx.sqrt(mx.sum(mx.square(increments), axis=-1) + 1e-12)
    directions = increments / increment_norms[..., None]
    residual_cosine = mx.mean(
        mx.sum(directions[:-1] * directions[1:], axis=-1)
    )
    residual_cv = mx.mean(
        mx.std(increment_norms, axis=0)
        / mx.maximum(mx.mean(increment_norms, axis=0), 1e-12)
    )
    aux = {
        "task_loss": task_loss,
        "penalty": penalty,
        "residual_adjacent_cosine": residual_cosine,
        "residual_norm_depth_cv": residual_cv,
        "residual_rms": mx.sqrt(mx.mean(mx.square(increments))),
        **terms,
    }
    return total_loss, aux


_PROJECTION_PATH = re.compile(
    r"^blocks\.(\d+)\.(attention\.(?:qkv|output)|mlp\.(?:up|down))\.(weight|bias)$"
)


def window_smooth(values: Sequence[mx.array], alpha: float) -> list[mx.array]:
    if len(values) < 2 or alpha == 0.0:
        return list(values)
    first = (1.0 - 0.5 * alpha) * values[0] + 0.5 * alpha * values[1]
    middle = [
        (1.0 - alpha) * values[index]
        + 0.5 * alpha * (values[index - 1] + values[index + 1])
        for index in range(1, len(values) - 1)
    ]
    last = (1.0 - 0.5 * alpha) * values[-1] + 0.5 * alpha * values[-2]
    return [first, *middle, last]


def _update_roughness(updates: dict[str, mx.array]) -> mx.array:
    groups: dict[str, list[tuple[int, mx.array]]] = {}
    for path, value in updates.items():
        match = _PROJECTION_PATH.match(path)
        if match:
            suffix = path.split(".", 2)[2]
            groups.setdefault(suffix, []).append((int(match.group(1)), value))
    difference_energy: list[mx.array] = []
    total_energy: list[mx.array] = []
    for entries in groups.values():
        values = [value for _, value in sorted(entries)]
        total_energy.extend(mx.sum(mx.square(value)) for value in values)
        difference_energy.extend(
            mx.sum(mx.square(right - left))
            for left, right in zip(values, values[1:])
        )
    return mx.sum(mx.stack(difference_energy)) / (
        mx.sum(mx.stack(total_energy)) + 1e-16
    )


class PaperAdamW:
    """Bias-corrected AdamW with paper-aligned Proj Gradient Smoothing."""

    def __init__(self, config: ExperimentConfig, smoothing: bool) -> None:
        self.learning_rate = config.learning_rate
        self.weight_decay = config.weight_decay
        self.alpha = config.smoothing_alpha if smoothing else 0.0
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8
        self.step = 0
        self.m: dict[str, mx.array] = {}
        self.v: dict[str, mx.array] = {}
        self.last_metrics: dict[str, float] = {}

    @property
    def state(self) -> dict[str, Any]:
        return {"m": self.m, "v": self.v}

    def update(self, model: TinyTransformer, gradients: dict[str, Any]) -> None:
        parameters = dict(tree_flatten(model.trainable_parameters()))
        flat_gradients = dict(tree_flatten(gradients))
        if not self.m:
            self.m = {path: mx.zeros_like(value) for path, value in parameters.items()}
            self.v = {path: mx.zeros_like(value) for path, value in parameters.items()}
        self.step += 1
        raw_updates: dict[str, mx.array] = {}
        for path, gradient in flat_gradients.items():
            self.m[path] = self.beta1 * self.m[path] + (1.0 - self.beta1) * gradient
            self.v[path] = self.beta2 * self.v[path] + (1.0 - self.beta2) * mx.square(
                gradient
            )
            m_hat = self.m[path] / (1.0 - self.beta1**self.step)
            v_hat = self.v[path] / (1.0 - self.beta2**self.step)
            raw_updates[path] = m_hat / (mx.sqrt(v_hat) + self.eps)

        applied_updates = dict(raw_updates)
        groups: dict[str, list[tuple[int, str, mx.array]]] = {}
        if self.alpha:
            for path, value in raw_updates.items():
                match = _PROJECTION_PATH.match(path)
                if match:
                    suffix = path.split(".", 2)[2]
                    groups.setdefault(suffix, []).append(
                        (int(match.group(1)), path, value)
                    )
            for entries in groups.values():
                entries.sort()
                smoothed = window_smooth([entry[2] for entry in entries], self.alpha)
                for (_, path, _), value in zip(entries, smoothed):
                    applied_updates[path] = value

        raw_roughness = _update_roughness(raw_updates)
        applied_roughness = _update_roughness(applied_updates)
        new_parameters = [
            (
                path,
                parameter * (1.0 - self.learning_rate * self.weight_decay)
                - self.learning_rate * applied_updates[path],
            )
            for path, parameter in parameters.items()
        ]
        model.update(tree_unflatten(new_parameters))
        mx.eval(model.parameters(), self.m, self.v, raw_roughness, applied_roughness)
        self.last_metrics = {
            "update_raw_roughness": float(raw_roughness.item()),
            "update_applied_roughness": float(applied_roughness.item()),
        }


def _robust_location_scale(values: Sequence[float], floor: float = 0.15) -> tuple[float, float]:
    if not values:
        return 0.0, 1.0
    center = statistics.median(values)
    mad = statistics.median(abs(value - center) for value in values)
    return center, max(1.4826 * mad, floor)


class FEEObserver:
    """Persistent-harm controller; every decision affects the next update.

    A propagation anomaly is necessary but deliberately insufficient.  FE-E is
    scheduled only when multiple propagation diagnostics remain abnormal and a
    fixed sentinel batch shows consecutive task damage.  This keeps useful
    learning transitions out of the intervention path.
    """

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.state = "CALIBRATION"
        self.reference_mass: float | None = None
        self.log_stiffness: list[float] = []
        self.log_gradient_norm: list[float] = []
        self.log_mass: list[float] = []
        self.coverage: list[float] = []
        self.propagation_abnormal_history: list[bool] = []
        self.sentinel_loss_history: list[float] = []
        self.sentinel_accuracy_history: list[float] = []
        self.damage_history: list[bool] = []
        self.damage_streak = 0
        self.active_remaining = 0
        self.recovery_remaining = 0
        self.force_probe = False
        self.last_score = 0.0
        self.last_reasons: list[str] = []
        self.last_components: dict[str, float] = {}
        self.last_loss: float | None = None
        self.training_loss_history: list[float] = []
        self.intervention_reference_mass: float | None = None
        self.intervention_count = 0
        self.escalation_level = 0
        self.current_gradient_ratio = config.gated_fee_gradient_ratio

    def _append_baseline(
        self,
        log_stiffness: float,
        log_gradient: float,
        log_mass: float,
        coverage: float,
    ) -> None:
        self.log_stiffness.append(log_stiffness)
        self.log_gradient_norm.append(log_gradient)
        self.log_mass.append(log_mass)
        self.coverage.append(coverage)
        window = self.config.observer_baseline_window
        self.log_stiffness = self.log_stiffness[-window:]
        self.log_gradient_norm = self.log_gradient_norm[-window:]
        self.log_mass = self.log_mass[-window:]
        self.coverage = self.coverage[-window:]

    def should_apply(self) -> bool:
        return self.active_remaining > 0

    def should_probe(self, step: int) -> bool:
        return (
            step < self.config.observer_calibration_steps
            and step % max(1, self.config.observer_probe_every // 2) == 0
        ) or step % self.config.observer_probe_every == 0 or self.force_probe

    def cheap_update(self, loss: float, gradient_norm: float) -> None:
        if self.last_loss is not None:
            relative_jump = (loss - self.last_loss) / max(abs(self.last_loss), 1e-8)
            if relative_jump > 0.15:
                self.force_probe = True
        self.last_loss = loss
        self.training_loss_history.append(loss)
        phase_horizon = (
            self.config.observer_probe_every
            * self.config.observer_persistence_window
        )
        if len(self.training_loss_history) > phase_horizon + 1:
            self.training_loss_history = self.training_loss_history[-(phase_horizon + 1) :]
        if self.log_gradient_norm:
            center, scale = _robust_location_scale(self.log_gradient_norm)
            if (math.log(gradient_norm + 1e-12) - center) / scale > 4.0:
                self.force_probe = True

    def observe(
        self,
        step: int,
        stiffness: float,
        mass: float,
        coverage: float,
        gradient_norm: float,
        sentinel_loss: float,
        sentinel_accuracy: float,
    ) -> dict[str, Any]:
        log_stiffness = math.log(stiffness + 1e-12)
        log_gradient = math.log(gradient_norm + 1e-12)
        log_mass = math.log(mass + 1e-12)
        if self.reference_mass is None:
            self.reference_mass = mass
        previous_sentinel_loss = (
            self.sentinel_loss_history[-1] if self.sentinel_loss_history else None
        )
        previous_sentinel_accuracy = (
            self.sentinel_accuracy_history[-1]
            if self.sentinel_accuracy_history
            else None
        )
        loss_relative_change = (
            (sentinel_loss - previous_sentinel_loss)
            / max(abs(previous_sentinel_loss), 1e-8)
            if previous_sentinel_loss is not None
            else 0.0
        )
        accuracy_change = (
            sentinel_accuracy - previous_sentinel_accuracy
            if previous_sentinel_accuracy is not None
            else 0.0
        )
        damage_event = bool(
            previous_sentinel_loss is not None
            and (
                loss_relative_change
                > self.config.observer_sentinel_loss_tolerance
                or accuracy_change
                < -self.config.observer_sentinel_accuracy_tolerance
            )
        )
        self.sentinel_loss_history.append(sentinel_loss)
        self.sentinel_accuracy_history.append(sentinel_accuracy)
        if step < self.config.observer_calibration_steps:
            self._append_baseline(
                log_stiffness, log_gradient, log_mass, coverage
            )
            self.state = "CALIBRATION"
            score = 0.0
            reasons = ["calibration"]
            score_map = {
                "stiffness_z": 0.0,
                "gradient_z": 0.0,
                "mass_z": 0.0,
                "coverage_deficit_z": 0.0,
            }
            metric_votes = 0
            propagation_abnormal = False
            persistent_propagation = False
            self.damage_streak = 0
            persistent_damage = False
            phase_improving = False
            phase_loss_change = 0.0
            confirmed_harm = False
        else:
            stiffness_center, stiffness_scale = _robust_location_scale(
                self.log_stiffness
            )
            gradient_center, gradient_scale = _robust_location_scale(
                self.log_gradient_norm
            )
            mass_center, mass_scale = _robust_location_scale(
                self.log_mass, floor=self.config.observer_mass_log_tolerance
            )
            coverage_center, coverage_scale = _robust_location_scale(
                self.coverage, floor=0.05
            )
            stiffness_z = max(0.0, (log_stiffness - stiffness_center) / stiffness_scale)
            gradient_z = max(0.0, (log_gradient - gradient_center) / gradient_scale)
            mass_score = abs(log_mass - mass_center) / mass_scale
            adaptive_coverage_floor = min(
                self.config.observer_coverage_floor,
                coverage_center - coverage_scale,
            )
            coverage_score = max(
                0.0, (adaptive_coverage_floor - coverage) / coverage_scale
            )
            score_map = {
                "stiffness_z": stiffness_z,
                "gradient_z": gradient_z,
                "mass_z": mass_score,
                "coverage_deficit_z": coverage_score,
            }
            score = max(score_map.values())
            reasons = [
                name
                for name, value in score_map.items()
                if value >= self.config.observer_on_threshold
            ]
            metric_votes = len(reasons)
            propagation_abnormal = (
                metric_votes >= self.config.observer_metric_votes_required
            )
            self.propagation_abnormal_history.append(propagation_abnormal)
            window = self.config.observer_persistence_window
            if len(self.propagation_abnormal_history) > window:
                self.propagation_abnormal_history = (
                    self.propagation_abnormal_history[-window:]
                )
            persistent_propagation = bool(
                len(self.propagation_abnormal_history) >= window
                and sum(self.propagation_abnormal_history)
                >= self.config.observer_persistence_required
            )
            self.damage_history.append(damage_event)
            if len(self.damage_history) > self.config.observer_harm_consecutive:
                self.damage_history = self.damage_history[
                    -self.config.observer_harm_consecutive :
                ]
            self.damage_streak = self.damage_streak + 1 if damage_event else 0
            persistent_damage = (
                self.damage_streak >= self.config.observer_harm_consecutive
            )
            phase_losses = (
                self.training_loss_history
                if len(self.training_loss_history) >= 2
                else self.sentinel_loss_history
            )
            phase_anchor_loss = phase_losses[0]
            phase_loss_change = (phase_losses[-1] - phase_anchor_loss) / max(
                abs(phase_anchor_loss), 1e-8
            )
            phase_improving = (
                phase_loss_change
                <= -self.config.observer_phase_improvement_tolerance
            )
            controller_busy = self.active_remaining > 0 or self.recovery_remaining > 0
            confirmed_harm = bool(
                propagation_abnormal
                and persistent_propagation
                and persistent_damage
                and not phase_improving
                and not controller_busy
            )
            if self.active_remaining > 0:
                self.state = "INTERVENE"
            elif self.recovery_remaining > 0:
                self.state = "COOLDOWN"
            elif confirmed_harm:
                self.active_remaining = self.config.intervention_steps
                self.intervention_reference_mass = math.exp(
                    statistics.median(self.log_mass)
                )
                self.recovery_remaining = 0
                self.current_gradient_ratio = min(
                    self.config.gated_fee_gradient_ratio
                    * (2**self.escalation_level),
                    self.config.gated_fee_gradient_ratio_max,
                )
                self.intervention_count += 1
                if self.current_gradient_ratio < self.config.gated_fee_gradient_ratio_max:
                    self.escalation_level += 1
                self.state = "CONFIRMED_HARM"
            elif score <= self.config.observer_off_threshold:
                self.state = "NORMAL"
            else:
                self.state = "WATCH"
            healthy_window = bool(
                len(self.propagation_abnormal_history) >= window
                and not any(self.propagation_abnormal_history[-window:])
                and self.damage_streak == 0
            )
            if healthy_window and not controller_busy:
                self.escalation_level = 0
            baseline_adapted = bool(
                self.config.observer_adaptive_baseline
                and not confirmed_harm
                and self.active_remaining == 0
                and not damage_event
            )
            if baseline_adapted:
                self._append_baseline(
                    log_stiffness, log_gradient, log_mass, coverage
                )
        if step < self.config.observer_calibration_steps:
            baseline_adapted = True
        self.force_probe = False
        self.last_score = score
        self.last_reasons = reasons
        self.last_components = {
            **score_map,
            "metric_votes": float(metric_votes),
            "persistent_abnormal_count": float(
                sum(self.propagation_abnormal_history)
            ),
            "sentinel_loss_relative_change": loss_relative_change,
            "sentinel_accuracy_change": accuracy_change,
            "damage_streak": float(self.damage_streak),
            "phase_loss_change": phase_loss_change,
        }
        return {
            "observer_score": score,
            "observer_reasons": reasons,
            "observer_components": self.last_components,
            "observer_decision_state": self.state,
            "observer_metric_votes": metric_votes,
            "observer_propagation_abnormal": propagation_abnormal,
            "observer_persistent_propagation": persistent_propagation,
            "observer_damage_event": damage_event,
            "observer_damage_streak": self.damage_streak,
            "observer_persistent_damage": persistent_damage,
            "observer_phase_improving": phase_improving,
            "observer_confirmed_harm": confirmed_harm,
            "observer_baseline_adapted": baseline_adapted,
            "observer_baseline_samples": len(self.log_stiffness),
            "observer_intervention_count": self.intervention_count,
            "observer_scheduled_gradient_ratio": self.current_gradient_ratio,
            "sentinel_loss": sentinel_loss,
            "sentinel_accuracy": sentinel_accuracy,
        }

    def advance_after_step(self, applied: bool) -> None:
        if applied and self.active_remaining > 0:
            self.active_remaining -= 1
            if self.active_remaining == 0:
                self.recovery_remaining = self.config.recovery_steps
                self.state = "COOLDOWN"
        elif self.recovery_remaining > 0:
            self.recovery_remaining -= 1
            if self.recovery_remaining == 0:
                self.state = "NORMAL"


class JsonlLogger:
    def __init__(self, path: Path, fsync_every: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.handle = path.open("w", encoding="utf-8", buffering=1)
        self.fsync_every = fsync_every
        self.count = 0

    def write(self, record: dict[str, Any]) -> None:
        self.handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        self.handle.flush()
        self.count += 1
        if self.count % self.fsync_every == 0:
            os.fsync(self.handle.fileno())

    def close(self) -> None:
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.handle.close()

    def __enter__(self) -> "JsonlLogger":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def _tokens(config: ExperimentConfig, seed: int, step: int) -> mx.array:
    key = mx.random.key(seed * 100_003 + step)
    return mx.random.randint(
        0,
        config.vocab_size,
        (config.batch_size, config.sequence_length),
        key=key,
    )


def _tree_l2_norm(tree: dict[str, Any]) -> mx.array:
    pieces = [mx.sum(mx.square(value)) for _, value in tree_flatten(tree)]
    return mx.sqrt(mx.sum(mx.stack(pieces)))


def _scalar(value: mx.array) -> float:
    return float(value.item())


def evaluate(
    model: TinyTransformer, config: ExperimentConfig, seed: int
) -> tuple[float, float]:
    losses: list[float] = []
    accuracies: list[float] = []
    for batch in range(config.evaluation_batches):
        tokens = _tokens(config, seed, 1_000_000 + batch)
        logits = model(tokens)
        loss = reverse_sequence_loss(logits, tokens)
        targets = tokens[:, ::-1]
        accuracy = mx.mean(mx.argmax(logits, axis=-1) == targets)
        mx.eval(loss, accuracy)
        losses.append(_scalar(loss))
        accuracies.append(_scalar(accuracy))
    return statistics.fmean(losses), statistics.fmean(accuracies)


def evaluate_sentinel(
    model: TinyTransformer, config: ExperimentConfig, seed: int
) -> tuple[float, float]:
    """Evaluate one immutable batch used only by the harm controller."""

    tokens = _tokens(config, seed, 2_000_000)
    logits = model(tokens)
    loss = reverse_sequence_loss(logits, tokens)
    accuracy = mx.mean(mx.argmax(logits, axis=-1) == tokens[:, ::-1])
    mx.eval(loss, accuracy)
    return _scalar(loss), _scalar(accuracy)


def _variant_flags(variant: str) -> tuple[bool, bool, bool]:
    smoothing = variant in {
        "gradient_smoothing",
        "gs_fe_e_gated",
        "gs_observer_control",
    }
    gated = variant in {"fe_e_gated", "gs_fe_e_gated"}
    monitored = gated or variant in {
        "adamw_observer_control",
        "gs_observer_control",
    }
    return smoothing, gated, monitored


def run_variant(
    config: ExperimentConfig,
    variant: str,
    seed: int,
    run_dir: Path,
) -> dict[str, Any]:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant}")
    mx.random.seed(seed)
    model = TinyTransformer(config)
    mx.eval(model.parameters())
    smoothing, gated, monitored = _variant_flags(variant)
    optimizer = PaperAdamW(config, smoothing=smoothing)
    observer = FEEObserver(config) if monitored else None
    reference_energy: float | None = None
    history_evaluations: list[dict[str, float | int]] = []
    step_times: list[float] = []
    timed_training_seconds = 0.0
    task_losses: list[float] = []
    gradient_norms: list[float] = []
    adjoint_metrics: list[dict[str, float | int]] = []
    regularized_steps = 0
    scheduled_intervention_steps = 0
    probe_steps = 0
    first_target_step: int | None = None
    target_streak_start_step: int | None = None
    target_streak = 0
    target_confirmed_step: int | None = None
    target_confirmed_streak_start_step: int | None = None
    target_confirmed_training_seconds: float | None = None
    target_confirmed_elapsed_seconds: float | None = None
    termination_reason = "maximum_steps"
    started = time.perf_counter()
    mx.reset_peak_memory()
    log_path = run_dir / "logs" / f"{variant}_seed{seed}.jsonl"

    loss_and_grad = nn.value_and_grad(model, loss_with_aux)
    status = "completed"
    error: str | None = None
    with JsonlLogger(log_path, config.log_fsync_every) as logger:
        logger.write(
            {
                "record_type": "run_start",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "variant": variant,
                "seed": seed,
                "config": asdict(config),
            }
        )
        for step in range(config.steps):
            step_started = time.perf_counter()
            step_cpu_started = time.process_time()
            stress_active = (
                config.stress_step >= 0
                and config.stress_step <= step < config.stress_step + config.stress_duration
            )
            optimizer.learning_rate = config.learning_rate * (
                config.stress_lr_multiplier if stress_active else 1.0
            )
            state_before = observer.state if observer else "NA"
            intervention_scheduled = bool(observer and observer.should_apply())
            if variant == "fe_e_always":
                apply_fee = True
            elif variant == "fe_e_periodic":
                apply_fee = step % config.periodic_every == 0
            elif gated:
                apply_fee = intervention_scheduled
            else:
                apply_fee = False
            scheduled_diagnostic = step % config.diagnostic_every == 0
            gate_probe = bool(observer and observer.should_probe(step))
            observer_probe = bool(
                observer and (gate_probe or intervention_scheduled)
            )
            compute_adjoint = (
                apply_fee
                or scheduled_diagnostic
                or gate_probe
                or step + 1 == config.steps
            )
            penalty_scale = (
                (config.intervention_steps - observer.active_remaining + 1)
                / config.intervention_steps
                if apply_fee and observer is not None
                else min(1.0, (step + 1) / config.regularizer_warmup_steps)
                if apply_fee and config.regularizer_warmup_steps > 0
                else float(apply_fee)
            )
            fee_reference_energy = (
                observer.intervention_reference_mass
                if apply_fee
                and observer is not None
                and observer.intervention_reference_mass is not None
                else reference_energy
            )
            tokens = _tokens(config, seed, step)
            task_gradient_norm_value: float | None = None
            fee_gradient_norm_value: float | None = None
            fee_gradient_scale = 1.0
            raw_combined_gradient_norm_value: float | None = None
            try:
                (total_loss, aux), gradients = loss_and_grad(
                    model,
                    tokens,
                    compute_adjoint,
                    apply_fee,
                    penalty_scale,
                    fee_reference_energy,
                    config,
                )
                if apply_fee and gated:
                    (_, _), task_gradients = loss_and_grad(
                        model,
                        tokens,
                        False,
                        False,
                        0.0,
                        None,
                        config,
                    )
                    fee_gradients = tree_map(
                        lambda combined, task: combined - task,
                        gradients,
                        task_gradients,
                    )
                    task_gradient_norm = _tree_l2_norm(task_gradients)
                    fee_gradient_norm = _tree_l2_norm(fee_gradients)
                    raw_combined_gradient_norm = _tree_l2_norm(gradients)
                    mx.eval(
                        task_gradients,
                        fee_gradients,
                        task_gradient_norm,
                        fee_gradient_norm,
                        raw_combined_gradient_norm,
                    )
                    task_gradient_norm_value = _scalar(task_gradient_norm)
                    fee_gradient_norm_value = _scalar(fee_gradient_norm)
                    raw_combined_gradient_norm_value = _scalar(
                        raw_combined_gradient_norm
                    )
                    fee_gradient_scale = min(
                        1.0,
                        (
                            observer.current_gradient_ratio
                            if observer is not None
                            else config.gated_fee_gradient_ratio
                        )
                        * task_gradient_norm_value
                        / max(fee_gradient_norm_value, 1e-12),
                    )
                    gradients = tree_map(
                        lambda task, fee: task + fee_gradient_scale * fee,
                        task_gradients,
                        fee_gradients,
                    )
                gradient_norm = _tree_l2_norm(gradients)
                mx.eval(total_loss, aux, gradients, gradient_norm)
                task_loss_value = _scalar(aux["task_loss"])
                gradient_norm_value = _scalar(gradient_norm)
                finite = math.isfinite(_scalar(total_loss)) and math.isfinite(
                    gradient_norm_value
                )
                if not finite:
                    raise FloatingPointError("non-finite loss or parameter gradient")
                if compute_adjoint and reference_energy is None:
                    reference_energy = _scalar(aux["mass_energy"])
                optimizer.update(model, gradients)
                mx.eval(model.parameters(), optimizer.state)
            except Exception as exc:
                status = "failed"
                termination_reason = "failure"
                error = f"{type(exc).__name__}: {exc}"
                logger.write(
                    {
                        "record_type": "failure",
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "variant": variant,
                        "seed": seed,
                        "step": step,
                        "error": error,
                    }
                )
                break

            observer_update: dict[str, Any] = {}
            sentinel_loss: float | None = None
            sentinel_accuracy: float | None = None
            if compute_adjoint:
                adjoint_metrics.append(
                    {
                        "step": step,
                        "stiffness_normalized": _scalar(aux["stiffness_normalized"]),
                        "mass_energy": _scalar(aux["mass_energy"]),
                        "relative_coverage": _scalar(aux["relative_coverage"]),
                    }
                )
            if observer:
                observer.cheap_update(task_loss_value, gradient_norm_value)
                if observer_probe:
                    sentinel_loss, sentinel_accuracy = evaluate_sentinel(
                        model, config, seed
                    )
                    observer_update = observer.observe(
                        step,
                        _scalar(aux["stiffness_normalized"]),
                        _scalar(aux["mass_energy"]),
                        _scalar(aux["relative_coverage"]),
                        gradient_norm_value,
                        sentinel_loss,
                        sentinel_accuracy,
                    )
                observer.advance_after_step(intervention_scheduled)

            optimization_seconds = time.perf_counter() - step_started
            optimization_cpu_seconds = time.process_time() - step_cpu_started
            step_times.append(optimization_seconds)
            if step >= config.timing_warmup_steps:
                timed_training_seconds += optimization_seconds
            evaluation_loss: float | None = None
            evaluation_accuracy: float | None = None
            evaluation_seconds = 0.0
            stop_after_step = False
            if (step + 1) % config.evaluation_every == 0 or step + 1 == config.steps:
                evaluation_started = time.perf_counter()
                evaluation_loss, evaluation_accuracy = evaluate(model, config, seed)
                evaluation_seconds = time.perf_counter() - evaluation_started
                if evaluation_accuracy >= config.target_token_accuracy:
                    if first_target_step is None:
                        first_target_step = step + 1
                    if target_streak == 0:
                        target_streak_start_step = step + 1
                    target_streak += 1
                    if (
                        target_streak >= config.target_confirmations
                        and target_confirmed_step is None
                    ):
                        target_confirmed_step = step + 1
                        target_confirmed_streak_start_step = target_streak_start_step
                        target_confirmed_training_seconds = timed_training_seconds
                        target_confirmed_elapsed_seconds = time.perf_counter() - started
                        if config.stop_on_target:
                            termination_reason = "target_confirmed"
                            stop_after_step = True
                else:
                    target_streak = 0
                    target_streak_start_step = None
                history_evaluations.append(
                    {
                        "step": step + 1,
                        "elapsed_seconds": time.perf_counter() - started,
                        "timed_training_seconds": timed_training_seconds,
                        "evaluation_loss": evaluation_loss,
                        "evaluation_accuracy": evaluation_accuracy,
                        "target_streak": target_streak,
                        "target_confirmed": target_confirmed_step is not None,
                    }
                )
            task_losses.append(task_loss_value)
            gradient_norms.append(gradient_norm_value)
            regularized_steps += int(apply_fee)
            scheduled_intervention_steps += int(intervention_scheduled)
            probe_steps += int(compute_adjoint and not apply_fee)
            record: dict[str, Any] = {
                "record_type": "train_step",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "variant": variant,
                "seed": seed,
                "step": step,
                "task_loss": task_loss_value,
                "total_loss": _scalar(total_loss),
                "evaluation_loss": evaluation_loss,
                "evaluation_accuracy": evaluation_accuracy,
                "parameter_gradient_norm": gradient_norm_value,
                "task_gradient_norm": task_gradient_norm_value,
                "fee_gradient_norm_raw": fee_gradient_norm_value,
                "fee_gradient_scale": fee_gradient_scale,
                "raw_combined_gradient_norm": raw_combined_gradient_norm_value,
                "effective_learning_rate": optimizer.learning_rate,
                "stress_active": stress_active,
                "finite": True,
                "regularized": apply_fee,
                "intervention_scheduled": intervention_scheduled,
                "adjoint_probe": compute_adjoint and not apply_fee,
                "observer_probe": observer_probe,
                "penalty_scale": penalty_scale,
                "step_seconds": optimization_seconds,
                "step_process_cpu_seconds": optimization_cpu_seconds,
                "evaluation_seconds": evaluation_seconds,
                "elapsed_seconds": time.perf_counter() - started,
                "timed_training_seconds": timed_training_seconds,
                "peak_memory_bytes": int(mx.get_peak_memory()),
                "observer_state_before": state_before,
                "observer_state_after": observer.state if observer else "NA",
                "observer_score": observer.last_score if observer else 0.0,
                "observer_reasons": observer.last_reasons if observer else [],
                "observer_components": observer.last_components if observer else {},
                "observer_active_remaining": observer.active_remaining if observer else 0,
                "observer_cooldown_remaining": (
                    observer.recovery_remaining if observer else 0
                ),
                "observer_gradient_ratio": (
                    observer.current_gradient_ratio if observer else 0.0
                ),
                "sentinel_loss": sentinel_loss,
                "sentinel_accuracy": sentinel_accuracy,
                "target_token_accuracy": config.target_token_accuracy,
                "target_streak": target_streak,
                "target_confirmed_step": target_confirmed_step,
                "system_load_average": list(os.getloadavg()),
                "global_reference_energy": reference_energy,
                "fee_reference_energy": fee_reference_energy,
                **optimizer.last_metrics,
                **observer_update,
            }
            for name, value in aux.items():
                if name != "task_loss":
                    record[name] = _scalar(value)
            logger.write(record)
            if step == 0 or (step + 1) % 25 == 0:
                print(
                    f"{variant:>18} seed={seed} step={step + 1:>3}/{config.steps} "
                    f"loss={task_loss_value:.4f} eval={evaluation_loss if evaluation_loss is not None else float('nan'):.4f} "
                    f"fee={int(apply_fee)} probe={int(compute_adjoint and not apply_fee)} "
                    f"{optimization_seconds:.3f}s",
                    flush=True,
                )
            if stop_after_step:
                print(
                    f"{variant:>18} seed={seed} target confirmed at step "
                    f"{target_confirmed_step} (streak started at "
                    f"{target_streak_start_step})",
                    flush=True,
                )
                break

        elapsed = time.perf_counter() - started
        logger.write(
            {
                "record_type": "run_end",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "variant": variant,
                "seed": seed,
                "status": status,
                "error": error,
                "termination_reason": termination_reason,
                "completed_steps": len(task_losses),
                "elapsed_seconds": elapsed,
                "timed_training_seconds": timed_training_seconds,
                "peak_memory_bytes": int(mx.get_peak_memory()),
            }
        )

    final_eval_loss, final_eval_accuracy = (
        (history_evaluations[-1]["evaluation_loss"], history_evaluations[-1]["evaluation_accuracy"])
        if history_evaluations
        else (float("nan"), float("nan"))
    )
    final_adjoint = adjoint_metrics[-1] if adjoint_metrics else {}
    result = {
        "variant": variant,
        "seed": seed,
        "status": status,
        "error": error,
        "termination_reason": termination_reason,
        "completed_steps": len(task_losses),
        "elapsed_seconds": elapsed,
        "mean_step_seconds": statistics.fmean(
            step_times[config.timing_warmup_steps :]
            if len(step_times) > config.timing_warmup_steps
            else step_times
        )
        if step_times
        else float("nan"),
        "timed_training_seconds": timed_training_seconds,
        "initial_task_loss": task_losses[0] if task_losses else float("nan"),
        "final_task_loss": task_losses[-1] if task_losses else float("nan"),
        "minimum_task_loss": min(task_losses) if task_losses else float("nan"),
        "max_parameter_gradient_norm": max(gradient_norms) if gradient_norms else float("nan"),
        "final_stiffness_normalized": final_adjoint.get(
            "stiffness_normalized", float("nan")
        ),
        "final_mass_energy": final_adjoint.get("mass_energy", float("nan")),
        "final_relative_coverage": final_adjoint.get(
            "relative_coverage", float("nan")
        ),
        "evaluation_loss": final_eval_loss,
        "evaluation_accuracy": final_eval_accuracy,
        "target_token_accuracy": config.target_token_accuracy,
        "target_confirmations_required": config.target_confirmations,
        "first_target_step": first_target_step,
        "target_streak_start_step": (
            target_confirmed_streak_start_step
        ),
        "target_confirmed_step": target_confirmed_step,
        "target_confirmed_training_seconds": target_confirmed_training_seconds,
        "target_confirmed_elapsed_seconds": target_confirmed_elapsed_seconds,
        "regularized_steps": regularized_steps,
        "regularized_fraction": regularized_steps / max(1, len(task_losses)),
        "scheduled_intervention_steps": scheduled_intervention_steps,
        "scheduled_intervention_fraction": (
            scheduled_intervention_steps / max(1, len(task_losses))
        ),
        "probe_steps": probe_steps,
        "probe_fraction": probe_steps / max(1, len(task_losses)),
        "observer_intervention_count": (
            observer.intervention_count if observer is not None else 0
        ),
        "peak_memory_bytes": int(mx.get_peak_memory()),
        "evaluation_history": history_evaluations,
        "log_path": str(log_path.relative_to(run_dir)),
    }
    result_path = run_dir / "runs" / f"{variant}_seed{seed}.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def _mean_sd(values: Iterable[float]) -> tuple[float, float]:
    items = list(values)
    return statistics.fmean(items), statistics.stdev(items) if len(items) > 1 else 0.0


def summarize(
    config: ExperimentConfig, results: Sequence[dict[str, Any]], run_dir: Path
) -> dict[str, Any]:
    successful = [result for result in results if result["status"] == "completed"]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for result in successful:
        grouped.setdefault(result["variant"], []).append(result)
    time_reference_variant = (
        "baseline"
        if grouped.get("baseline")
        else successful[0]["variant"] if successful else None
    )
    time_reference_times = {
        result["seed"]: result["mean_step_seconds"]
        for result in grouped.get(time_reference_variant, [])
    }
    baseline_budgets = {
        result["seed"]: result["timed_training_seconds"]
        for result in grouped.get("baseline", [])
    }
    methods: dict[str, Any] = {}
    for variant, items in grouped.items():
        aggregate: dict[str, Any] = {"n": len(items)}
        for metric in (
            "evaluation_loss",
            "evaluation_accuracy",
            "mean_step_seconds",
            "regularized_fraction",
            "probe_fraction",
            "peak_memory_bytes",
            "max_parameter_gradient_norm",
            "final_stiffness_normalized",
            "final_mass_energy",
            "final_relative_coverage",
        ):
            mean, sd = _mean_sd(float(item[metric]) for item in items)
            aggregate[metric] = {"mean": mean, "sd": sd}
        ratios = [
            item["mean_step_seconds"] / time_reference_times[item["seed"]]
            for item in items
            if item["seed"] in time_reference_times
        ]
        mean, sd = _mean_sd(ratios)
        aggregate["time_ratio"] = {"mean": mean, "sd": sd}
        budget_losses: list[float] = []
        budget_steps: list[int] = []
        for item in items:
            budget = baseline_budgets.get(item["seed"])
            eligible = [
                checkpoint
                for checkpoint in item["evaluation_history"]
                if budget is not None
                and checkpoint["timed_training_seconds"] <= budget
            ]
            if eligible:
                budget_losses.append(float(eligible[-1]["evaluation_loss"]))
                budget_steps.append(int(eligible[-1]["step"]))
        if budget_losses:
            mean, sd = _mean_sd(budget_losses)
            aggregate["baseline_training_time_loss"] = {"mean": mean, "sd": sd}
            aggregate["baseline_training_time_steps"] = statistics.fmean(budget_steps)
        confirmed_items = [
            item for item in items if item.get("target_confirmed_step") is not None
        ]
        aggregate["target_confirmation_rate"] = len(confirmed_items) / len(items)
        for metric in (
            "first_target_step",
            "target_streak_start_step",
            "target_confirmed_step",
            "target_confirmed_training_seconds",
            "target_confirmed_elapsed_seconds",
        ):
            values = [float(item[metric]) for item in confirmed_items]
            if values:
                mean, sd = _mean_sd(values)
                aggregate[metric] = {"mean": mean, "sd": sd}
        methods[variant] = aggregate

    paired: dict[str, Any] = {}
    baseline_by_seed = {
        item["seed"]: item for item in grouped.get("baseline", [])
    }
    for variant, items in grouped.items():
        if variant == "baseline":
            continue
        differences = [
            item["evaluation_loss"] - baseline_by_seed[item["seed"]]["evaluation_loss"]
            for item in items
            if item["seed"] in baseline_by_seed
        ]
        if differences:
            mean, sd = _mean_sd(differences)
            paired[variant] = {"loss_difference_mean": mean, "loss_difference_sd": sd}
    summary = {
        "config": asdict(config),
        "methods": methods,
        "paired_vs_baseline": paired,
        "time_reference_variant": time_reference_variant,
        "successful_runs": len(successful),
        "total_runs": len(results),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    labels = {
        "baseline": "AdamW",
        "gradient_smoothing": "Gradient Smoothing",
        "fe_e_always": "FE-E 常开",
        "fe_e_periodic": f"FE-E 固定每 {config.periodic_every} 步",
        "fe_e_gated": "观测器门控 FE-E",
        "gs_fe_e_gated": "GS + 门控 FE-E",
        "adamw_observer_control": "AdamW（同构观察器）",
        "gs_observer_control": "纯 GS（同构观察器）",
    }
    rows = [
        f"# MLX {config.layers} 层 FE-E 架构对比",
        "",
        f"Apple MLX；{config.layers} 层、宽度 {config.width}、{config.steps} 步。",
        "",
        f"主终点：token accuracy ≥ {config.target_token_accuracy:.1%}，连续 "
        f"{config.target_confirmations} 个验证检查点确认。",
        "",
        f"| 方法 | 达标率 | 首次跨线步 | 确认步 | 训练秒至确认 | 评估损失 ↓ | 准确率 ↑ | 时间/{labels.get(time_reference_variant, '参考')} | FE-E 介入率 | 探测率 | 峰值内存 MiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        if variant not in methods:
            continue
        method = methods[variant]
        loss = method["evaluation_loss"]
        accuracy = method["evaluation_accuracy"]
        ratio = method["time_ratio"]
        regularized = method["regularized_fraction"]
        probe = method["probe_fraction"]
        stiffness = method["final_stiffness_normalized"]
        coverage = method["final_relative_coverage"]
        memory = method["peak_memory_bytes"]
        first_target = method.get("first_target_step")
        confirmed = method.get("target_confirmed_step")
        target_seconds = method.get("target_confirmed_training_seconds")
        first_target_text = f"{first_target['mean']:.0f}" if first_target else "—"
        confirmed_text = f"{confirmed['mean']:.0f}" if confirmed else "—"
        seconds_text = f"{target_seconds['mean']:.1f}" if target_seconds else "—"
        rows.append(
            f"| {labels[variant]} | {method['target_confirmation_rate']:.0%} | "
            f"{first_target_text} | {confirmed_text} | {seconds_text} | "
            f"{loss['mean']:.4f} ± {loss['sd']:.4f} | "
            f"{accuracy['mean']:.3f} ± {accuracy['sd']:.3f} | "
            f"{ratio['mean']:.2f} ± {ratio['sd']:.2f}× | "
            f"{regularized['mean']:.1%} | {probe['mean']:.1%} | "
            f"{memory['mean'] / 2**20:.1f} |"
        )
    rows.extend(
        [
            "",
            "逐步原始记录位于 `logs/*.jsonl`；每次运行摘要位于 `runs/*.json`；完整环境见 `manifest.json`。",
            "",
            "说明：这是合成反序列任务上的工程筛选实验，不等同于语言模型预训练结论。",
        ]
    )
    (run_dir / "summary.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return summary


def _command_output(command: list[str]) -> str | None:
    try:
        return subprocess.run(
            command, check=True, capture_output=True, text=True
        ).stdout.strip()
    except Exception:
        return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment_manifest(config: ExperimentConfig, argv: Sequence[str]) -> dict[str, Any]:
    source = Path(__file__).resolve()
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, "-m", "fe_entropy.mlx_experiment", *argv],
        "config": asdict(config),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "mlx_version": importlib.metadata.version("mlx"),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "macos_version": _command_output(["sw_vers", "-productVersion"]),
        "hardware_model": _command_output(["sysctl", "-n", "hw.model"]),
        "physical_memory_bytes": _command_output(["sysctl", "-n", "hw.memsize"]),
        "mlx_device": mx.device_info(),
        "source_file": str(source),
        "source_sha256": _sha256(source),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--layers", type=int, default=24)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--lambda-stiffness", type=float, default=2.0)
    parser.add_argument("--lambda-energy", type=float, default=0.02)
    parser.add_argument("--lambda-entropy", type=float, default=2.0)
    parser.add_argument("--entropy-lower", type=float, default=0.90)
    parser.add_argument("--entropy-upper", type=float, default=0.98)
    parser.add_argument("--seeds", default="31,47,59")
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--periodic-every", type=int, default=8)
    parser.add_argument("--diagnostic-every", type=int, default=20)
    parser.add_argument("--observer-probe-every", type=int, default=8)
    parser.add_argument("--evaluation-every", type=int, default=25)
    parser.add_argument("--evaluation-batches", type=int, default=10)
    parser.add_argument("--observer-calibration-steps", type=int, default=24)
    parser.add_argument("--observer-on-threshold", type=float, default=2.5)
    parser.add_argument("--observer-persistence-window", type=int, default=4)
    parser.add_argument("--observer-persistence-required", type=int, default=3)
    parser.add_argument("--observer-metric-votes-required", type=int, default=2)
    parser.add_argument("--observer-harm-consecutive", type=int, default=2)
    parser.add_argument("--observer-sentinel-loss-tolerance", type=float, default=0.01)
    parser.add_argument(
        "--observer-sentinel-accuracy-tolerance", type=float, default=0.01
    )
    parser.add_argument(
        "--observer-phase-improvement-tolerance", type=float, default=0.02
    )
    parser.add_argument("--observer-baseline-window", type=int, default=16)
    parser.add_argument("--observer-adaptive-baseline", action="store_true")
    parser.add_argument("--intervention-steps", type=int, default=1)
    parser.add_argument("--recovery-steps", type=int, default=48)
    parser.add_argument("--stress-step", type=int, default=-1)
    parser.add_argument("--stress-duration", type=int, default=8)
    parser.add_argument("--stress-lr-multiplier", type=float, default=3.0)
    parser.add_argument("--gated-fee-gradient-ratio", type=float, default=0.05)
    parser.add_argument("--gated-fee-gradient-ratio-max", type=float, default=0.20)
    parser.add_argument("--target-token-accuracy", type=float, default=0.99)
    parser.add_argument("--target-confirmations", type=int, default=3)
    parser.add_argument("--stop-on-target", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("results/mlx_d24"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config = ExperimentConfig(
        steps=args.steps,
        layers=args.layers,
        width=args.width,
        heads=args.heads,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lambda_stiffness=args.lambda_stiffness,
        lambda_energy=args.lambda_energy,
        lambda_entropy=args.lambda_entropy,
        entropy_lower=args.entropy_lower,
        entropy_upper=args.entropy_upper,
        periodic_every=args.periodic_every,
        diagnostic_every=args.diagnostic_every,
        observer_probe_every=args.observer_probe_every,
        evaluation_every=args.evaluation_every,
        evaluation_batches=args.evaluation_batches,
        observer_calibration_steps=args.observer_calibration_steps,
        observer_on_threshold=args.observer_on_threshold,
        observer_persistence_window=args.observer_persistence_window,
        observer_persistence_required=args.observer_persistence_required,
        observer_metric_votes_required=args.observer_metric_votes_required,
        observer_harm_consecutive=args.observer_harm_consecutive,
        observer_sentinel_loss_tolerance=args.observer_sentinel_loss_tolerance,
        observer_sentinel_accuracy_tolerance=(
            args.observer_sentinel_accuracy_tolerance
        ),
        observer_phase_improvement_tolerance=(
            args.observer_phase_improvement_tolerance
        ),
        observer_baseline_window=args.observer_baseline_window,
        observer_adaptive_baseline=args.observer_adaptive_baseline,
        intervention_steps=args.intervention_steps,
        recovery_steps=args.recovery_steps,
        stress_step=args.stress_step,
        stress_duration=args.stress_duration,
        stress_lr_multiplier=args.stress_lr_multiplier,
        gated_fee_gradient_ratio=args.gated_fee_gradient_ratio,
        gated_fee_gradient_ratio_max=args.gated_fee_gradient_ratio_max,
        target_token_accuracy=args.target_token_accuracy,
        target_confirmations=args.target_confirmations,
        stop_on_target=args.stop_on_target,
    )
    config.validate()
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    unknown = set(variants) - set(VARIANTS)
    if unknown:
        raise ValueError(f"unknown variants: {sorted(unknown)}")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / f"run_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest = environment_manifest(config, sys.argv[1:] if argv is None else argv)
    manifest["seeds"] = seeds
    manifest["variants"] = variants
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    results: list[dict[str, Any]] = []
    for seed in seeds:
        for variant in variants:
            result = run_variant(config, variant, seed, run_dir)
            results.append(result)
            summarize(config, results, run_dir)
            if result["status"] != "completed":
                print(
                    f"FAILED {variant} seed={seed}: {result['error']}",
                    file=sys.stderr,
                    flush=True,
                )
    summary = summarize(config, results, run_dir)
    print(f"\nResults: {run_dir.resolve()}")
    for variant, metrics in summary["methods"].items():
        print(
            f"{variant:>18}: loss={metrics['evaluation_loss']['mean']:.4f} "
            f"time={metrics['time_ratio']['mean']:.2f}x "
            f"fee={metrics['regularized_fraction']['mean']:.1%}",
            flush=True,
        )


if __name__ == "__main__":
    main()
