"""ICML 2026 Gradient Smoothing applied to AdamW optimizer updates.

The implementation follows Algorithm 1 and Appendix A.1 of
"Gradient Smoothing: Coupling Layer-wise Updates for Improved Optimization".
In particular, smoothing is applied *after* Adam moment preconditioning and
decoupled weight decay is kept outside the smoothing operator.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import torch
from torch import Tensor, nn
from torch.optim import Optimizer


def window_smooth_tensors(values: Sequence[Tensor], alpha: float) -> list[Tensor]:
    """Apply the paper's symmetric row-stochastic tridiagonal window."""

    if not 0.0 <= alpha < 1.0:
        raise ValueError("alpha must lie in [0, 1)")
    if not values:
        raise ValueError("values cannot be empty")
    if len(values) == 1 or alpha == 0.0:
        return [value.clone() for value in values]
    if any(value.shape != values[0].shape for value in values[1:]):
        raise ValueError("corresponding block updates must have equal shapes")

    smoothed = [
        (1.0 - alpha) * value
        + 0.5 * alpha * (values[index - 1] + values[index + 1])
        for index, value in enumerate(values[1:-1], start=1)
    ]
    first = (1.0 - 0.5 * alpha) * values[0] + 0.5 * alpha * values[1]
    last = (1.0 - 0.5 * alpha) * values[-1] + 0.5 * alpha * values[-2]
    return [first, *smoothed, last]


def _block_vector_norm(
    update_map: dict[nn.Parameter, Tensor], parameters: Iterable[nn.Parameter]
) -> Tensor:
    pieces = [update_map[parameter].square().sum() for parameter in parameters]
    if not pieces:
        raise ValueError("a smoothed block must contain parameters")
    return torch.sqrt(torch.stack(pieces).sum())


def _block_update_metrics(
    updates: dict[nn.Parameter, Tensor],
    block_maps: Sequence[dict[str, nn.Parameter]],
    names: Sequence[str],
    eps: float = 1e-16,
) -> dict[str, float]:
    norms: list[Tensor] = []
    for block_map in block_maps:
        norms.append(
            _block_vector_norm(updates, (block_map[name] for name in names))
        )
    energy = torch.stack([norm.square() for norm in norms]).sum()
    difference_energy: list[Tensor] = []
    cosine: list[Tensor] = []
    for left_index in range(len(block_maps) - 1):
        left_map = block_maps[left_index]
        right_map = block_maps[left_index + 1]
        difference_energy.append(
            torch.stack(
                [
                    (updates[right_map[name]] - updates[left_map[name]])
                    .square()
                    .sum()
                    for name in names
                ]
            ).sum()
        )
        dot = torch.stack(
            [
                (updates[left_map[name]] * updates[right_map[name]]).sum()
                for name in names
            ]
        ).sum()
        cosine.append(dot / (norms[left_index] * norms[left_index + 1] + eps))
    roughness = torch.stack(difference_energy).sum() / (energy + eps)
    return {
        "roughness": float(roughness.cpu()),
        "adjacent_cosine": float(torch.stack(cosine).mean().cpu()),
        "mean_block_norm": float(torch.stack(norms).mean().cpu()),
    }


class GradientSmoothingAdamW(Optimizer):
    """AdamW with depth-wise window smoothing of repeated-block updates.

    ``scope='proj'`` applies smoothing only to parameters owned by linear
    modules, matching the paper's Proj configuration. ``variant`` implements
    Standard, Norm-Preserving, and Directional smoothing from Section 3.4.
    """

    def __init__(
        self,
        params: Iterable[nn.Parameter],
        *,
        blocks: Sequence[nn.Module],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        alpha: float = 0.1,
        variant: str = "standard",
        scope: str = "proj",
    ) -> None:
        if lr < 0.0:
            raise ValueError("lr must be non-negative")
        if eps < 0.0:
            raise ValueError("eps must be non-negative")
        if weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError("betas must lie in [0, 1)")
        if not 0.0 <= alpha < 1.0:
            raise ValueError("alpha must lie in [0, 1)")
        if variant not in {"standard", "norm", "directional"}:
            raise ValueError("variant must be standard, norm, or directional")
        if scope not in {"full", "proj"}:
            raise ValueError("scope must be full or proj")
        block_list = list(blocks)
        if len(block_list) < 2:
            raise ValueError("Gradient Smoothing requires at least two blocks")

        parameter_list = list(params)
        super().__init__(
            parameter_list,
            dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay),
        )
        self.alpha = alpha
        self.smoothing_variant = variant
        self.smoothing_scope = scope
        self._block_maps = self._build_block_maps(block_list, scope)
        self._selected_names = tuple(self._block_maps[0])
        self.last_metrics: dict[str, float] = {}

        all_ids = {id(parameter) for parameter in parameter_list}
        selected_ids = {
            id(block_map[name])
            for block_map in self._block_maps
            for name in self._selected_names
        }
        if not selected_ids <= all_ids:
            raise ValueError("all block parameters must be present in params")

    @staticmethod
    def _build_block_maps(
        blocks: Sequence[nn.Module], scope: str
    ) -> list[dict[str, nn.Parameter]]:
        maps: list[dict[str, nn.Parameter]] = []
        for block in blocks:
            parameters = dict(block.named_parameters())
            if scope == "proj":
                linear_modules = {
                    name for name, module in block.named_modules() if isinstance(module, nn.Linear)
                }
                parameters = {
                    name: parameter
                    for name, parameter in parameters.items()
                    if name.rpartition(".")[0] in linear_modules
                }
            if not parameters:
                raise ValueError("no parameters selected for smoothing")
            maps.append(parameters)

        expected_names = tuple(maps[0])
        for block_map in maps[1:]:
            if tuple(block_map) != expected_names:
                raise ValueError("repeated blocks must expose identical parameter structure")
            for name in expected_names:
                if block_map[name].shape != maps[0][name].shape:
                    raise ValueError("corresponding block parameters must have equal shapes")
        return maps

    def _smooth_updates(
        self, updates: dict[nn.Parameter, Tensor]
    ) -> dict[nn.Parameter, Tensor]:
        # Skip the allocation-heavy path for the exact AdamW baseline.
        if self.alpha == 0.0:
            return updates

        block_parameters = [
            [block_map[name] for name in self._selected_names]
            for block_map in self._block_maps
        ]
        original_norms = [
            _block_vector_norm(updates, parameters) for parameters in block_parameters
        ]

        if self.smoothing_variant == "directional":
            source = {
                parameter: updates[parameter] / original_norms[index].clamp_min(1e-16)
                for index, parameters in enumerate(block_parameters)
                for parameter in parameters
            }
        else:
            source = updates

        result = dict(updates)
        for name in self._selected_names:
            values = [source[block_map[name]] for block_map in self._block_maps]
            for block_map, value in zip(
                self._block_maps,
                window_smooth_tensors(values, self.alpha),
                strict=True,
            ):
                result[block_map[name]] = value

        if self.smoothing_variant in {"norm", "directional"}:
            smoothed_norms = [
                _block_vector_norm(result, parameters)
                for parameters in block_parameters
            ]
            for index, parameters in enumerate(block_parameters):
                scale = original_norms[index] / smoothed_norms[index].clamp_min(1e-16)
                for parameter in parameters:
                    result[parameter] = result[parameter] * scale
        return result

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        group = self.param_groups[0]
        beta1, beta2 = group["betas"]
        updates: dict[nn.Parameter, Tensor] = {}
        for parameter in group["params"]:
            gradient = parameter.grad
            if gradient is None:
                continue
            if gradient.is_sparse:
                raise RuntimeError("GradientSmoothingAdamW does not support sparse gradients")
            state = self.state[parameter]
            if not state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(parameter)
                state["exp_avg_sq"] = torch.zeros_like(parameter)
            state["step"] += 1
            step = state["step"]
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]
            exp_avg.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(gradient, gradient, value=1.0 - beta2)
            bias_correction1 = 1.0 - beta1**step
            bias_correction2 = 1.0 - beta2**step
            denominator = exp_avg_sq.sqrt() / math.sqrt(bias_correction2)
            denominator.add_(group["eps"])
            updates[parameter] = (exp_avg / bias_correction1) / denominator

        missing = [
            parameter
            for block_map in self._block_maps
            for parameter in block_map.values()
            if parameter not in updates
        ]
        if missing:
            raise RuntimeError("all selected block parameters must receive gradients")

        raw_metrics = _block_update_metrics(
            updates, self._block_maps, self._selected_names
        )
        smoothed = self._smooth_updates(updates)
        smoothed_metrics = _block_update_metrics(
            smoothed, self._block_maps, self._selected_names
        )
        self.last_metrics = {
            f"update_raw_{name}": value for name, value in raw_metrics.items()
        } | {
            f"update_applied_{name}": value
            for name, value in smoothed_metrics.items()
        }

        for parameter, update in smoothed.items():
            parameter.mul_(1.0 - group["lr"] * group["weight_decay"])
            parameter.add_(update, alpha=-group["lr"])
        return loss

