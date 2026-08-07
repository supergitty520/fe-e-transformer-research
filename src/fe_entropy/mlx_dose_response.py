"""Depth-noise dose-response experiment for GS and FE-E on Apple MLX.

The experiment is deliberately an intervention-rate instrument, not a
production controller.  Four compute-matched GS trajectories are compared in
each propagation environment: a diagnostic sham and nested, frozen-random
FE-E schedules targeting 1%, 3%, and 5% of optimizer steps.  Every trajectory
runs through the learning phase transition until token accuracy is at least
99% for three consecutive validation checkpoints (or a safety cap is hit).

Every optimizer step is synchronously appended to JSONL for paper audit and
failure recovery.  Validation uses eight immutable synthetic batches and is
always clean, including while structured training noise is active.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import time
from typing import Any, Sequence

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map

from .mlx_experiment import (
    ExperimentConfig,
    JsonlLogger,
    PaperAdamW,
    TinyTransformer,
    _scalar,
    _sha256,
    _tokens,
    _tree_l2_norm,
    environment_manifest,
    evaluate,
    fe_terms,
    fee_penalty,
    reverse_sequence_loss,
)


NOISE_MODES = ("none", "high_frequency", "energy", "concentration")
VARIANTS = ("gs_sham", "gsf_q01", "gsf_q03", "gsf_q05")
DOSE_BY_VARIANT = {
    "gs_sham": 0.00,
    "gsf_q01": 0.01,
    "gsf_q03": 0.03,
    "gsf_q05": 0.05,
}
MILESTONE_THRESHOLDS = (0.10, 0.50, 0.90, 0.99)


def intervention_schedules(
    max_steps: int,
    schedule_seed: int,
) -> dict[str, frozenset[int]]:
    """Build nested frozen-random 1/3/5-per-100 schedules.

    Step zero is reserved for establishing the common FE energy reference.
    Each full 100-step block nevertheless contains the exact requested dose.
    """

    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    selected: list[int] = []
    for block_start in range(0, max_steps, 100):
        block_stop = min(block_start + 100, max_steps)
        candidates = list(range(block_start, block_stop))
        if block_start == 0 and 0 in candidates:
            candidates.remove(0)
        requested = min(5, len(candidates))
        generator = random.Random(schedule_seed + 1_000_003 * (block_start // 100))
        block_positions = generator.sample(candidates, requested)
        selected.extend(block_positions)
    q01 = frozenset(selected[index] for index in range(0, len(selected), 5))
    q03 = frozenset(
        selected[index]
        for index in range(len(selected))
        if index % 5 < 3
    )
    q05 = frozenset(selected)
    return {
        "gs_sham": frozenset(),
        "gsf_q01": q01,
        "gsf_q03": q03,
        "gsf_q05": q05,
    }


def residual_profile(layers: int, mode: str) -> mx.array:
    """Return the per-block residual-increment multiplier for one noise mode."""

    if layers <= 0:
        raise ValueError("layers must be positive")
    if mode == "none":
        values = [1.0] * layers
    elif mode == "high_frequency":
        values = [1.25 if index % 2 == 0 else 0.75 for index in range(layers)]
    elif mode == "energy":
        values = [1.5] * layers
    elif mode == "concentration":
        concentrated = max(1, layers // 8)
        start = (layers - concentrated) // 2
        stop = start + concentrated
        outside = (layers - concentrated * 2.5) / (layers - concentrated)
        values = [
            2.5 if start <= index < stop else outside
            for index in range(layers)
        ]
    else:
        raise ValueError(f"unknown noise mode: {mode}")
    return mx.array(values)


def active_profile(
    layers: int,
    mode: str,
    step: int,
    noise_start: int,
    noise_duration: int,
) -> tuple[mx.array, bool]:
    active = bool(
        mode != "none" and noise_start <= step < noise_start + noise_duration
    )
    return residual_profile(layers, mode if active else "none"), active


def profiled_block(
    hidden: mx.array,
    block: nn.Module,
    scale: mx.array,
) -> mx.array:
    candidate = block(hidden)
    return hidden + scale * (candidate - hidden)


def forward_states_profiled(
    model: TinyTransformer,
    tokens: mx.array,
    profile: mx.array,
) -> list[mx.array]:
    hidden = model.token_embedding(tokens) + model.position_embedding[: tokens.shape[1]]
    states = [hidden]
    for index, block in enumerate(model.blocks):
        hidden = profiled_block(hidden, block, profile[index])
        states.append(hidden)
    return states


def hidden_adjoint_field_profiled(
    model: TinyTransformer,
    states: Sequence[mx.array],
    tokens: mx.array,
    profile: mx.array,
) -> list[mx.array]:
    """Return exact task-loss adjoints for the actually profiled propagation."""

    def head_loss(hidden: mx.array) -> mx.array:
        return reverse_sequence_loss(model.logits_from_hidden(hidden), tokens)

    gradient = mx.grad(head_loss)(states[-1])
    reversed_gradients = [gradient]
    for index in range(len(model.blocks) - 1, -1, -1):
        block = model.blocks[index]
        scale = profile[index]

        def transition(hidden: mx.array) -> mx.array:
            return profiled_block(hidden, block, scale)

        _, vjps = mx.vjp(transition, [states[index]], [gradient])
        gradient = vjps[0]
        reversed_gradients.append(gradient)
    return list(reversed(reversed_gradients))


def _zero_terms() -> dict[str, mx.array]:
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


def loss_with_aux_profiled(
    model: TinyTransformer,
    tokens: mx.array,
    profile: mx.array,
    compute_adjoint: bool,
    apply_fee: bool,
    reference_energy: float | None,
    config: ExperimentConfig,
) -> tuple[mx.array, dict[str, mx.array]]:
    states = forward_states_profiled(model, tokens, profile)
    logits = model.logits_from_hidden(states[-1])
    task_loss = reverse_sequence_loss(logits, tokens)
    if compute_adjoint:
        terms = fe_terms(hidden_adjoint_field_profiled(model, states, tokens, profile))
        penalty, penalty_terms = fee_penalty(terms, reference_energy, config)
        terms = {**terms, **penalty_terms}
    else:
        terms = _zero_terms()
        penalty = mx.array(0.0)
    total_loss = task_loss + penalty if apply_fee else task_loss
    increments = mx.stack(
        [right - left for left, right in zip(states, states[1:])]
    )
    increment_norms = mx.sqrt(mx.sum(mx.square(increments), axis=-1) + 1e-12)
    directions = increments / increment_norms[..., None]
    aux = {
        "task_loss": task_loss,
        "penalty": penalty,
        "residual_adjacent_cosine": mx.mean(
            mx.sum(directions[:-1] * directions[1:], axis=-1)
        ),
        "residual_norm_depth_cv": mx.mean(
            mx.std(increment_norms, axis=0)
            / mx.maximum(mx.mean(increment_norms, axis=0), 1e-12)
        ),
        "residual_rms": mx.sqrt(mx.mean(mx.square(increments))),
        **terms,
    }
    return total_loss, aux


def _tree_dot(left: dict[str, Any], right: dict[str, Any]) -> mx.array:
    left_flat = dict(tree_flatten(left))
    right_flat = dict(tree_flatten(right))
    return mx.sum(
        mx.stack(
            [mx.sum(left_flat[path] * right_flat[path]) for path in left_flat]
        )
    )


def _parameter_delta_norm(
    before: dict[str, mx.array],
    after: dict[str, mx.array],
) -> mx.array:
    return mx.sqrt(
        mx.sum(
            mx.stack(
                [mx.sum(mx.square(after[path] - before[path])) for path in before]
            )
        )
    )


def _milestone_key(threshold: float) -> str:
    return f"first_accuracy_{int(round(100 * threshold)):02d}_step"


def _noise_metadata(mode: str, layers: int) -> dict[str, Any]:
    if mode == "none":
        return {"description": "unit residual profile"}
    if mode == "high_frequency":
        return {"description": "alternating residual profile", "factors": [1.25, 0.75]}
    if mode == "energy":
        return {"description": "global residual-energy amplification", "factor": 1.5}
    concentrated = max(1, layers // 8)
    outside = (layers - concentrated * 2.5) / (layers - concentrated)
    return {
        "description": "middle-layer residual concentration at unit mean",
        "middle_layers": concentrated,
        "middle_factor": 2.5,
        "outside_factor": outside,
    }


def run_trajectory(
    config: ExperimentConfig,
    seed: int,
    noise_mode: str,
    variant: str,
    schedules: dict[str, frozenset[int]],
    schedule_seed: int,
    noise_start: int,
    noise_duration: int,
    diagnostic_every: int,
    fee_gradient_ratio: float,
    run_dir: Path,
) -> dict[str, Any]:
    if noise_mode not in NOISE_MODES:
        raise ValueError(f"unknown noise mode: {noise_mode}")
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant: {variant}")
    if not 0.0 < fee_gradient_ratio <= 1.0:
        raise ValueError("fee_gradient_ratio must lie in (0, 1]")

    mx.random.seed(seed)
    model = TinyTransformer(config)
    mx.eval(model.parameters())
    optimizer = PaperAdamW(config, smoothing=True)
    loss_and_grad = nn.value_and_grad(model, loss_with_aux_profiled)
    reference_energy: float | None = None
    intervention_steps = schedules[variant]
    diagnostic_union = schedules["gsf_q05"]
    milestones = {_milestone_key(value): None for value in MILESTONE_THRESHOLDS}
    target_streak = 0
    target_streak_start: int | None = None
    target_confirmed_step: int | None = None
    target_confirmed_streak_start: int | None = None
    histories: list[dict[str, Any]] = []
    task_losses: list[float] = []
    step_times: list[float] = []
    intervention_count = 0
    diagnostic_count = 0
    started = time.perf_counter()
    timed_training_seconds = 0.0
    termination_reason = "maximum_steps"
    status = "completed"
    error: str | None = None
    mx.reset_peak_memory()
    stem = f"{noise_mode}_{variant}_seed{seed}"
    log_path = run_dir / "logs" / f"{stem}.jsonl"

    with JsonlLogger(log_path, config.log_fsync_every) as logger:
        logger.write(
            {
                "record_type": "run_start",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "noise_mode": noise_mode,
                "variant": variant,
                "seed": seed,
                "target_intervention_rate": DOSE_BY_VARIANT[variant],
                "schedule_seed": schedule_seed,
                "noise_start_zero_based": noise_start,
                "noise_duration": noise_duration,
                "config": asdict(config),
            }
        )
        for step in range(config.steps):
            step_started = time.perf_counter()
            cpu_started = time.process_time()
            profile, noise_active = active_profile(
                config.layers, noise_mode, step, noise_start, noise_duration
            )
            planned_intervention = step in intervention_steps
            compute_adjoint = bool(
                step == 0
                or step % diagnostic_every == 0
                or step in diagnostic_union
                or step + 1 == config.steps
            )
            tokens = _tokens(config, seed, step)
            task_gradient_norm_value: float | None = None
            fee_gradient_norm_value: float | None = None
            fee_gradient_scale = 0.0
            fee_task_cosine: float | None = None
            fee_task_dot: float | None = None
            try:
                (total_loss, aux), gradients = loss_and_grad(
                    model,
                    tokens,
                    profile,
                    compute_adjoint,
                    planned_intervention,
                    reference_energy,
                    config,
                )
                if planned_intervention:
                    (_, _), task_gradients = loss_and_grad(
                        model,
                        tokens,
                        profile,
                        False,
                        False,
                        None,
                        config,
                    )
                    fee_gradients = tree_map(
                        lambda combined, task: combined - task,
                        gradients,
                        task_gradients,
                    )
                    task_norm = _tree_l2_norm(task_gradients)
                    fee_norm = _tree_l2_norm(fee_gradients)
                    task_fee_dot = _tree_dot(task_gradients, fee_gradients)
                    mx.eval(task_gradients, fee_gradients, task_norm, fee_norm, task_fee_dot)
                    task_gradient_norm_value = _scalar(task_norm)
                    fee_gradient_norm_value = _scalar(fee_norm)
                    fee_task_dot = _scalar(task_fee_dot)
                    fee_task_cosine = fee_task_dot / max(
                        task_gradient_norm_value * fee_gradient_norm_value, 1e-12
                    )
                    fee_gradient_scale = min(
                        1.0,
                        fee_gradient_ratio
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
                if compute_adjoint and reference_energy is None:
                    reference_energy = _scalar(aux["mass_energy"])
                task_loss_value = _scalar(aux["task_loss"])
                total_loss_value = _scalar(total_loss)
                gradient_norm_value = _scalar(gradient_norm)
                if not all(
                    math.isfinite(value)
                    for value in (task_loss_value, total_loss_value, gradient_norm_value)
                ):
                    raise FloatingPointError("non-finite loss or parameter gradient")
                parameters_before = dict(tree_flatten(model.trainable_parameters()))
                optimizer.update(model, gradients)
                parameters_after = dict(tree_flatten(model.trainable_parameters()))
                update_delta_norm = _parameter_delta_norm(
                    parameters_before, parameters_after
                )
                mx.eval(update_delta_norm, model.parameters(), optimizer.state)
                update_delta_norm_value = _scalar(update_delta_norm)
            except Exception as exc:
                status = "failed"
                termination_reason = "failure"
                error = f"{type(exc).__name__}: {exc}"
                logger.write(
                    {
                        "record_type": "failure",
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "noise_mode": noise_mode,
                        "variant": variant,
                        "seed": seed,
                        "step": step,
                        "error": error,
                    }
                )
                break

            optimization_seconds = time.perf_counter() - step_started
            optimization_cpu_seconds = time.process_time() - cpu_started
            step_times.append(optimization_seconds)
            if step >= config.timing_warmup_steps:
                timed_training_seconds += optimization_seconds
            task_losses.append(task_loss_value)
            intervention_count += int(planned_intervention)
            diagnostic_count += int(compute_adjoint)

            evaluation_loss: float | None = None
            evaluation_accuracy: float | None = None
            evaluation_seconds = 0.0
            stop_after_step = False
            if (step + 1) % config.evaluation_every == 0:
                evaluation_started = time.perf_counter()
                evaluation_loss, evaluation_accuracy = evaluate(model, config, seed)
                evaluation_seconds = time.perf_counter() - evaluation_started
                for threshold in MILESTONE_THRESHOLDS:
                    key = _milestone_key(threshold)
                    if milestones[key] is None and evaluation_accuracy >= threshold:
                        milestones[key] = step + 1
                if evaluation_accuracy >= config.target_token_accuracy:
                    if target_streak == 0:
                        target_streak_start = step + 1
                    target_streak += 1
                    if target_streak >= config.target_confirmations:
                        target_confirmed_step = step + 1
                        target_confirmed_streak_start = target_streak_start
                        termination_reason = "target_confirmed"
                        stop_after_step = True
                else:
                    target_streak = 0
                    target_streak_start = None
                histories.append(
                    {
                        "step": step + 1,
                        "evaluation_loss": evaluation_loss,
                        "evaluation_accuracy": evaluation_accuracy,
                        "target_streak": target_streak,
                        "noise_active": noise_active,
                        "elapsed_seconds": time.perf_counter() - started,
                        "timed_training_seconds": timed_training_seconds,
                    }
                )

            record: dict[str, Any] = {
                "record_type": "train_step",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "noise_mode": noise_mode,
                "variant": variant,
                "seed": seed,
                "step": step,
                "optimizer_step": step + 1,
                "task_loss": task_loss_value,
                "total_loss": total_loss_value,
                "evaluation_loss": evaluation_loss,
                "evaluation_accuracy": evaluation_accuracy,
                "target_streak": target_streak,
                "target_confirmed_step": target_confirmed_step,
                "planned_intervention": planned_intervention,
                "regularized": planned_intervention,
                "target_intervention_rate": DOSE_BY_VARIANT[variant],
                "realized_intervention_count": intervention_count,
                "realized_intervention_rate": intervention_count / (step + 1),
                "compute_adjoint": compute_adjoint,
                "diagnostic_count": diagnostic_count,
                "diagnostic_rate": diagnostic_count / (step + 1),
                "parameter_gradient_norm": gradient_norm_value,
                "task_gradient_norm": task_gradient_norm_value,
                "fee_gradient_norm_raw": fee_gradient_norm_value,
                "fee_gradient_scale": fee_gradient_scale,
                "fee_applied_to_task_gradient_ratio": (
                    fee_gradient_scale * fee_gradient_norm_value
                    / max(task_gradient_norm_value, 1e-12)
                    if task_gradient_norm_value is not None
                    and fee_gradient_norm_value is not None
                    else None
                ),
                "fee_task_gradient_dot": fee_task_dot,
                "fee_task_gradient_cosine": fee_task_cosine,
                "parameter_update_delta_norm": update_delta_norm_value,
                "noise_active": noise_active,
                "residual_profile_mean": _scalar(mx.mean(profile)),
                "residual_profile_min": _scalar(mx.min(profile)),
                "residual_profile_max": _scalar(mx.max(profile)),
                "step_seconds": optimization_seconds,
                "step_process_cpu_seconds": optimization_cpu_seconds,
                "evaluation_seconds": evaluation_seconds,
                "elapsed_seconds": time.perf_counter() - started,
                "timed_training_seconds": timed_training_seconds,
                "peak_memory_bytes": int(mx.get_peak_memory()),
                "global_reference_energy": reference_energy,
                "system_load_average": list(os.getloadavg()),
                **milestones,
                **optimizer.last_metrics,
            }
            for name, value in aux.items():
                if name != "task_loss":
                    record[name] = _scalar(value)
            logger.write(record)

            if step == 0 or (step + 1) % 100 == 0 or evaluation_accuracy is not None:
                accuracy_text = (
                    f"{evaluation_accuracy:.3f}" if evaluation_accuracy is not None else "---"
                )
                print(
                    f"{noise_mode:>14}/{variant:<8} step={step + 1:>4}/{config.steps} "
                    f"loss={task_loss_value:.4f} acc={accuracy_text} "
                    f"fee={intervention_count}/{step + 1} {optimization_seconds:.3f}s",
                    flush=True,
                )
            if stop_after_step:
                break

        elapsed = time.perf_counter() - started
        logger.write(
            {
                "record_type": "run_end",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "noise_mode": noise_mode,
                "variant": variant,
                "seed": seed,
                "status": status,
                "error": error,
                "termination_reason": termination_reason,
                "completed_steps": len(task_losses),
                "target_confirmed_step": target_confirmed_step,
                "elapsed_seconds": elapsed,
                "timed_training_seconds": timed_training_seconds,
                "peak_memory_bytes": int(mx.get_peak_memory()),
            }
        )

    first_10 = milestones[_milestone_key(0.10)]
    first_90 = milestones[_milestone_key(0.90)]
    result = {
        "noise_mode": noise_mode,
        "variant": variant,
        "seed": seed,
        "status": status,
        "error": error,
        "termination_reason": termination_reason,
        "completed_steps": len(task_losses),
        "target_intervention_rate": DOSE_BY_VARIANT[variant],
        "realized_intervention_count": intervention_count,
        "realized_intervention_rate": intervention_count / max(1, len(task_losses)),
        "diagnostic_count": diagnostic_count,
        "diagnostic_rate": diagnostic_count / max(1, len(task_losses)),
        "initial_task_loss": task_losses[0] if task_losses else None,
        "final_task_loss": task_losses[-1] if task_losses else None,
        "minimum_task_loss": min(task_losses) if task_losses else None,
        "final_evaluation_loss": histories[-1]["evaluation_loss"] if histories else None,
        "final_evaluation_accuracy": (
            histories[-1]["evaluation_accuracy"] if histories else None
        ),
        **milestones,
        "transition_width_10_90_steps": (
            first_90 - first_10 if first_10 is not None and first_90 is not None else None
        ),
        "target_streak_start_step": target_confirmed_streak_start,
        "target_confirmed_step": target_confirmed_step,
        "mean_step_seconds": (
            statistics.fmean(step_times[config.timing_warmup_steps :])
            if len(step_times) > config.timing_warmup_steps
            else statistics.fmean(step_times)
            if step_times
            else None
        ),
        "timed_training_seconds": timed_training_seconds,
        "elapsed_seconds": elapsed,
        "peak_memory_bytes": int(mx.get_peak_memory()),
        "evaluation_history": histories,
        "log_path": str(log_path.relative_to(run_dir)),
    }
    result_path = run_dir / "runs" / f"{stem}.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def write_partial_summary(results: Sequence[dict[str, Any]], run_dir: Path) -> None:
    payload = {
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "completed_trajectories": len(results),
        "results": list(results),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=8000)
    parser.add_argument("--layers", type=int, default=128)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--schedule-seed", type=int, default=20260806)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--noise-start", type=int, default=128)
    parser.add_argument("--noise-duration", type=int, default=500)
    parser.add_argument("--diagnostic-every", type=int, default=20)
    parser.add_argument("--evaluation-every", type=int, default=32)
    parser.add_argument("--evaluation-batches", type=int, default=8)
    parser.add_argument("--fee-gradient-ratio", type=float, default=0.05)
    parser.add_argument("--noise-modes", default=",".join(NOISE_MODES))
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/mlx_d128_s47_fee_dose_noise_acc99_eval8"),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config = ExperimentConfig(
        steps=args.max_steps,
        layers=args.layers,
        width=args.width,
        heads=args.heads,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        lambda_stiffness=0.1,
        lambda_energy=0.02,
        lambda_entropy=2.0,
        entropy_lower=0.50,
        entropy_upper=0.98,
        diagnostic_every=args.diagnostic_every,
        evaluation_every=args.evaluation_every,
        evaluation_batches=args.evaluation_batches,
        smoothing_alpha=0.20,
        target_token_accuracy=0.99,
        target_confirmations=3,
        stop_on_target=True,
    )
    config.validate()
    if args.noise_start < 0 or args.noise_duration <= 0:
        raise ValueError("noise interval must have non-negative start and positive duration")
    noise_modes = [value.strip() for value in args.noise_modes.split(",") if value.strip()]
    variants = [value.strip() for value in args.variants.split(",") if value.strip()]
    unknown_noise = set(noise_modes) - set(NOISE_MODES)
    unknown_variants = set(variants) - set(VARIANTS)
    if unknown_noise:
        raise ValueError(f"unknown noise modes: {sorted(unknown_noise)}")
    if unknown_variants:
        raise ValueError(f"unknown variants: {sorted(unknown_variants)}")

    schedules = intervention_schedules(config.steps, args.schedule_seed)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / f"run_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    command_args = list(sys.argv[1:] if argv is None else argv)
    manifest = environment_manifest(config, command_args)
    manifest.update(
        {
            "command": [sys.executable, "-m", "fe_entropy.mlx_dose_response", *command_args],
            "experiment_module": "fe_entropy.mlx_dose_response",
            "experiment_source_file": str(Path(__file__).resolve()),
            "experiment_source_sha256": _sha256(Path(__file__).resolve()),
            "seed": args.seed,
            "schedule_seed": args.schedule_seed,
            "noise_modes": noise_modes,
            "variants": variants,
            "noise_start_zero_based": args.noise_start,
            "noise_duration": args.noise_duration,
            "noise_definitions": {
                mode: _noise_metadata(mode, config.layers) for mode in noise_modes
            },
            "fee_gradient_ratio": args.fee_gradient_ratio,
            "schedule_counts": {
                variant: len(schedules[variant]) for variant in VARIANTS
            },
            "schedules": {
                variant: sorted(schedules[variant]) for variant in VARIANTS
            },
            "phase_transition_endpoint": {
                "validation_every_optimizer_steps": config.evaluation_every,
                "fixed_validation_batches": config.evaluation_batches,
                "target_token_accuracy": config.target_token_accuracy,
                "consecutive_confirmations": config.target_confirmations,
                "safety_cap_steps": config.steps,
                "milestones": list(MILESTONE_THRESHOLDS),
            },
        }
    )
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    results: list[dict[str, Any]] = []
    for noise_mode in noise_modes:
        for variant in variants:
            result = run_trajectory(
                config=config,
                seed=args.seed,
                noise_mode=noise_mode,
                variant=variant,
                schedules=schedules,
                schedule_seed=args.schedule_seed,
                noise_start=args.noise_start,
                noise_duration=args.noise_duration,
                diagnostic_every=args.diagnostic_every,
                fee_gradient_ratio=args.fee_gradient_ratio,
                run_dir=run_dir,
            )
            results.append(result)
            write_partial_summary(results, run_dir)
            if result["status"] != "completed":
                print(
                    f"FAILED {noise_mode}/{variant}: {result['error']}",
                    file=sys.stderr,
                    flush=True,
                )
    print(f"Results: {run_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
