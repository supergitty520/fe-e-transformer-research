"""Reproducible synthetic experiment for exact FE-E regularization."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import statistics
import time

import torch

from .gradient_smoothing import GradientSmoothingAdamW
from .model import TinyTransformer, reverse_sequence_loss
from .regularizer import FEERegularizer


@dataclass
class ExperimentConfig:
    steps: int = 30
    layers: int = 12
    width: int = 32
    heads: int = 4
    sequence_length: int = 12
    batch_size: int = 8
    vocab_size: int = 32
    learning_rate: float = 0.001
    seed: int = 7
    diagnostic_every: int = 5
    regularize_every: int = 1
    lambda_stiffness: float = 1e-4
    lambda_energy: float = 1e-2
    lambda_entropy: float = 1e-2
    residual_scale: float = 1.0
    depth_domain: str = "layer_index"
    entropy_lower: float = 0.65
    entropy_upper: float = 0.98
    evaluation_batches: int = 10
    regularizer_warmup_steps: int = 0
    evaluation_every: int = 0
    weight_decay: float = 0.01
    smoothing_alpha: float = 0.1
    smoothing_scope: str = "proj"

    def __post_init__(self) -> None:
        for name in (
            "steps",
            "layers",
            "width",
            "heads",
            "sequence_length",
            "batch_size",
            "vocab_size",
            "diagnostic_every",
            "regularize_every",
            "evaluation_batches",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.regularizer_warmup_steps < 0:
            raise ValueError("regularizer_warmup_steps cannot be negative")
        if self.evaluation_every < 0:
            raise ValueError("evaluation_every cannot be negative")


def _tokens(config: ExperimentConfig, step: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(config.seed * 100_003 + step)
    return torch.randint(
        config.vocab_size,
        (config.batch_size, config.sequence_length),
        generator=generator,
    )


def _parameter_gradient_norm(model: torch.nn.Module) -> float:
    squared = [
        parameter.grad.detach().square().sum()
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    return float(torch.sqrt(torch.stack(squared).sum())) if squared else 0.0


def _regularizer(config: ExperimentConfig, variant: str) -> FEERegularizer:
    if variant in {
        "baseline",
        "gradient_smoothing",
        "gradient_smoothing_norm",
        "gradient_smoothing_directional",
    }:
        coefficients = (0.0, 0.0, 0.0)
    elif variant == "stiffness":
        coefficients = (config.lambda_stiffness, 0.0, 0.0)
    elif variant == "energy":
        coefficients = (0.0, config.lambda_energy, 0.0)
    elif variant == "entropy":
        coefficients = (0.0, 0.0, config.lambda_entropy)
    elif variant == "fe":
        coefficients = (
            config.lambda_stiffness,
            config.lambda_energy,
            0.0,
        )
    elif variant in {"fe_e", "fe_entropy"}:
        coefficients = (
            config.lambda_stiffness,
            config.lambda_energy,
            config.lambda_entropy,
        )
    elif variant == "norm_proxy":
        return FEERegularizer(
            lambda_stiffness=config.lambda_stiffness,
            lambda_energy=config.lambda_energy,
            lambda_entropy=config.lambda_entropy,
            stiffness_mode="norm",
            entropy_band=(config.entropy_lower, config.entropy_upper),
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")
    return FEERegularizer(
        lambda_stiffness=coefficients[0],
        lambda_energy=coefficients[1],
        lambda_entropy=coefficients[2],
        entropy_band=(config.entropy_lower, config.entropy_upper),
    )


def _optimizer(
    config: ExperimentConfig, variant: str, model: TinyTransformer
) -> GradientSmoothingAdamW:
    alpha = config.smoothing_alpha if variant.startswith("gradient_smoothing") else 0.0
    smoothing_variant = {
        "gradient_smoothing_norm": "norm",
        "gradient_smoothing_directional": "directional",
    }.get(variant, "standard")
    return GradientSmoothingAdamW(
        model.parameters(),
        blocks=model.blocks,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        alpha=alpha,
        variant=smoothing_variant,
        scope=config.smoothing_scope,
    )


def _depth_positions(config: ExperimentConfig) -> torch.Tensor:
    if config.depth_domain == "layer_index":
        return torch.arange(config.layers + 1, dtype=torch.float32)
    if config.depth_domain == "unit_interval":
        return torch.linspace(0.0, 1.0, config.layers + 1)
    raise ValueError("depth_domain must be 'layer_index' or 'unit_interval'")


@torch.no_grad()
def _representation_metrics(states: list[torch.Tensor]) -> dict[str, float]:
    increments = torch.stack(
        [right.detach() - left.detach() for left, right in zip(states, states[1:])]
    )
    norms = torch.linalg.vector_norm(increments, dim=-1)
    directions = increments / norms.clamp_min(1e-12).unsqueeze(-1)
    adjacent_cosine = (directions[:-1] * directions[1:]).sum(dim=-1).mean()
    line_shape = increments.shape[0] / torch.linalg.vector_norm(
        directions.sum(dim=0), dim=-1
    ).clamp_min(1e-12)
    depth_cv = norms.std(dim=0, correction=0) / norms.mean(dim=0).clamp_min(1e-12)
    return {
        "residual_adjacent_cosine": float(adjacent_cosine.cpu()),
        "residual_line_shape_score": float(line_shape.mean().cpu()),
        "residual_norm_depth_cv": float(depth_cv.mean().cpu()),
        "residual_rms": float(torch.sqrt(increments.square().mean()).cpu()),
    }


@torch.no_grad()
def _evaluate(
    config: ExperimentConfig, model: TinyTransformer
) -> tuple[float, float]:
    was_training = model.training
    model.eval()
    losses: list[float] = []
    accuracies: list[float] = []
    for evaluation_step in range(config.evaluation_batches):
        tokens = _tokens(config, 1_000_000 + evaluation_step)
        logits, _ = model(tokens)
        losses.append(float(reverse_sequence_loss(logits, tokens)))
        targets = torch.flip(tokens, dims=(1,))
        accuracies.append(float((logits.argmax(dim=-1) == targets).float().mean()))
    model.train(was_training)
    return statistics.fmean(losses), statistics.fmean(accuracies)


def run_variant(config: ExperimentConfig, variant: str) -> dict[str, object]:
    torch.manual_seed(config.seed)
    model = TinyTransformer(
        vocab_size=config.vocab_size,
        sequence_length=config.sequence_length,
        layers=config.layers,
        width=config.width,
        heads=config.heads,
        residual_scale=config.residual_scale,
    )
    regularizer = _regularizer(config, variant)
    optimizer = _optimizer(config, variant, model)
    positions = _depth_positions(config)
    history: list[dict[str, float | int | bool]] = []
    evaluation_history: list[dict[str, float | int]] = []
    started = time.perf_counter()

    for step in range(config.steps):
        step_started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        tokens = _tokens(config, step)
        logits, states = model(tokens)
        task_loss = reverse_sequence_loss(logits, tokens)
        apply_regularizer = regularizer.is_active and step % config.regularize_every == 0
        collect_diagnostics = apply_regularizer or (
            not regularizer.is_active and step % config.diagnostic_every == 0
        )

        metrics: dict[str, float] = {}
        penalty_scale = 1.0
        if collect_diagnostics:
            output = regularizer(
                task_loss,
                states,
                positions=positions,
                create_graph=apply_regularizer,
            )
            metrics = output.metrics()
            if config.regularizer_warmup_steps > 0:
                penalty_scale = min(
                    1.0, (step + 1) / config.regularizer_warmup_steps
                )
            total_loss = (
                task_loss + penalty_scale * output.penalty
                if apply_regularizer
                else task_loss
            )
        else:
            total_loss = task_loss

        if step % config.diagnostic_every == 0:
            metrics.update(_representation_metrics(states))

        total_loss.backward()
        parameter_gradient_norm = _parameter_gradient_norm(model)
        finite = bool(
            torch.isfinite(total_loss).item() and math.isfinite(parameter_gradient_norm)
        )
        if finite:
            optimizer.step()
            metrics.update(optimizer.last_metrics)

        record: dict[str, float | int | bool] = {
            "step": step,
            "task_loss": float(task_loss.detach()),
            "total_loss": float(total_loss.detach()),
            "parameter_gradient_norm": parameter_gradient_norm,
            "finite": finite,
            "regularized": apply_regularizer,
            "penalty_scale": penalty_scale if apply_regularizer else 0.0,
            "step_seconds": time.perf_counter() - step_started,
        }
        record.update(metrics)
        history.append(record)
        if not finite:
            break
        if config.evaluation_every and (step + 1) % config.evaluation_every == 0:
            checkpoint_loss, checkpoint_accuracy = _evaluate(config, model)
            evaluation_history.append(
                {
                    "step": step + 1,
                    "elapsed_seconds": time.perf_counter() - started,
                    "evaluation_loss": checkpoint_loss,
                    "evaluation_accuracy": checkpoint_accuracy,
                }
            )

    elapsed = time.perf_counter() - started
    evaluation_loss, evaluation_accuracy = _evaluate(config, model)
    task_losses = [float(item["task_loss"]) for item in history]
    gradient_norms = [float(item["parameter_gradient_norm"]) for item in history]
    return {
        "variant": variant,
        "completed_steps": len(history),
        "all_finite": all(bool(item["finite"]) for item in history),
        "elapsed_seconds": elapsed,
        "mean_step_seconds": statistics.fmean(
            float(item["step_seconds"]) for item in history
        ),
        "initial_task_loss": task_losses[0],
        "final_task_loss": task_losses[-1],
        "minimum_task_loss": min(task_losses),
        "max_parameter_gradient_norm": max(gradient_norms),
        "evaluation_loss": evaluation_loss,
        "evaluation_accuracy": evaluation_accuracy,
        "evaluation_history": evaluation_history,
        "history": history,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--lambda-stiffness", type=float, default=1e-4)
    parser.add_argument("--lambda-energy", type=float, default=1e-2)
    parser.add_argument("--lambda-entropy", type=float, default=1e-2)
    parser.add_argument("--entropy-lower", type=float, default=0.65)
    parser.add_argument("--entropy-upper", type=float, default=0.98)
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--regularize-every", type=int, default=1)
    parser.add_argument("--diagnostic-every", type=int, default=5)
    parser.add_argument("--evaluation-every", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--smoothing-alpha", type=float, default=0.1)
    parser.add_argument(
        "--smoothing-scope", choices=("proj", "full"), default="proj"
    )
    parser.add_argument(
        "--depth-domain",
        choices=("layer_index", "unit_interval"),
        default="layer_index",
        help="Use layer_index for ordinary blocks; unit_interval only with an O(1/L) residual step",
    )
    parser.add_argument(
        "--variants",
        default="baseline,fe,fe_e,norm_proxy",
        help="Comma-separated subset of baseline,gradient_smoothing,gradient_smoothing_norm,gradient_smoothing_directional,stiffness,energy,entropy,fe,fe_e,norm_proxy (fe_entropy is a legacy alias)",
    )
    parser.add_argument("--output", type=Path, default=Path("results/experiment.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig(
        steps=args.steps,
        layers=args.layers,
        width=args.width,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        learning_rate=args.learning_rate,
        seed=args.seed,
        lambda_stiffness=args.lambda_stiffness,
        lambda_energy=args.lambda_energy,
        lambda_entropy=args.lambda_entropy,
        residual_scale=args.residual_scale,
        depth_domain=args.depth_domain,
        entropy_lower=args.entropy_lower,
        entropy_upper=args.entropy_upper,
        regularizer_warmup_steps=args.warmup_steps,
        regularize_every=args.regularize_every,
        diagnostic_every=args.diagnostic_every,
        evaluation_every=args.evaluation_every,
        weight_decay=args.weight_decay,
        smoothing_alpha=args.smoothing_alpha,
        smoothing_scope=args.smoothing_scope,
    )
    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    payload = {
        "config": asdict(config),
        "torch_version": torch.__version__,
        "results": [run_variant(config, variant) for variant in variants],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    for result in payload["results"]:
        print(
            f"{result['variant']:>10}  "
            f"loss {result['initial_task_loss']:.4f} -> {result['final_task_loss']:.4f}  "
            f"eval {result['evaluation_loss']:.4f}/{result['evaluation_accuracy']:.3f}  "
            f"step {result['mean_step_seconds']:.3f}s  "
            f"finite={result['all_finite']}"
        )


if __name__ == "__main__":
    main()
