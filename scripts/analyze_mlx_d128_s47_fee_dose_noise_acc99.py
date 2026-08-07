#!/usr/bin/env python3
"""Audit and visualize the 128-layer FE-E dose-response phase experiment."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
from typing import Any
from xml.sax.saxutils import escape

from PIL import Image, ImageDraw, ImageFont


NOISE_MODES = ("none", "high_frequency", "energy", "concentration")
VARIANTS = ("gs_sham", "gsf_q01", "gsf_q03", "gsf_q05")
LABELS = {
    "gs_sham": "GS-SHAM (0%)",
    "gsf_q01": "GSF 1%",
    "gsf_q03": "GSF 3%",
    "gsf_q05": "GSF 5%",
}
NOISE_LABELS = {
    "none": "No propagation noise",
    "high_frequency": "Alternating depth-frequency noise",
    "energy": "Global residual-energy amplification",
    "concentration": "Middle-layer energy concentration",
}
COLORS = {
    "gs_sham": "#202124",
    "gsf_q01": "#1f77b4",
    "gsf_q03": "#e08b19",
    "gsf_q05": "#c83e4d",
}
MARKERS = {"gs_sham": "o", "gsf_q01": "s", "gsf_q03": "^", "gsf_q05": "D"}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def sustained_onset(history: list[dict[str, Any]], low: float, high: float) -> int | None:
    high_index = next(
        (index for index, row in enumerate(history) if row["evaluation_accuracy"] >= high),
        None,
    )
    if high_index is None:
        return None
    last_below = max(
        (
            index
            for index, row in enumerate(history[: high_index + 1])
            if row["evaluation_accuracy"] < low
        ),
        default=-1,
    )
    return int(history[last_below + 1]["step"])


def first_step(history: list[dict[str, Any]], threshold: float) -> int | None:
    row = next(
        (row for row in history if row["evaluation_accuracy"] >= threshold),
        None,
    )
    return int(row["step"]) if row else None


def fixed_horizon_accuracy_deficit(
    history: list[dict[str, Any]], horizon: int, evaluation_every: int
) -> float:
    by_step = {int(row["step"]): float(row["evaluation_accuracy"]) for row in history}
    values: list[float] = []
    last_accuracy = 0.0
    for step in range(evaluation_every, horizon + 1, evaluation_every):
        if step in by_step:
            last_accuracy = by_step[step]
        elif last_accuracy >= 0.99:
            last_accuracy = 1.0
        values.append(1.0 - last_accuracy)
    return statistics.fmean(values)


def loss_slope_before_transition(
    train_rows: list[dict[str, Any]], onset: int | None
) -> float | None:
    if onset is None:
        return None
    start = max(0, onset - 256)
    stop = max(start + 2, onset - 32)
    selected = [
        row for row in train_rows if start <= int(row["optimizer_step"]) <= stop
    ]
    if len(selected) < 2:
        return None
    x = [float(row["optimizer_step"]) for row in selected]
    y = [float(row["task_loss"]) for row in selected]
    mean_x = statistics.fmean(x)
    mean_y = statistics.fmean(y)
    denominator = sum((value - mean_x) ** 2 for value in x)
    if denominator == 0.0:
        return None
    slope = sum(
        (x_value - mean_x) * (y_value - mean_y)
        for x_value, y_value in zip(x, y)
    ) / denominator
    return slope * 100.0


def median_or_none(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def audit_and_analyze(run_dir: Path) -> dict[str, Any]:
    manifest = load_json(run_dir / "manifest.json")
    summary = load_json(run_dir / "summary.json")
    result_by_key = {
        (row["noise_mode"], row["variant"]): row for row in summary["results"]
    }
    expected_keys = {(noise, variant) for noise in NOISE_MODES for variant in VARIANTS}
    if set(result_by_key) != expected_keys:
        raise AssertionError(
            f"trajectory set mismatch: missing={sorted(expected_keys-set(result_by_key))} "
            f"extra={sorted(set(result_by_key)-expected_keys)}"
        )
    schedules = {
        variant: set(steps) for variant, steps in manifest["schedules"].items()
    }
    if not schedules["gsf_q01"] < schedules["gsf_q03"] < schedules["gsf_q05"]:
        raise AssertionError("manifest intervention schedules are not strictly nested")

    audit_rows: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    for noise_mode in NOISE_MODES:
        for variant in VARIANTS:
            result = result_by_key[(noise_mode, variant)]
            records = load_jsonl(run_dir / result["log_path"])
            starts = [row for row in records if row["record_type"] == "run_start"]
            trains = [row for row in records if row["record_type"] == "train_step"]
            ends = [row for row in records if row["record_type"] == "run_end"]
            failures = [row for row in records if row["record_type"] == "failure"]
            expected_count = int(result["completed_steps"])
            contiguous = [row["step"] for row in trains] == list(range(expected_count))
            actual_interventions = {
                int(row["step"]) for row in trains if row["regularized"]
            }
            expected_interventions = {
                step for step in schedules[variant] if step < expected_count
            }
            eval_steps = [
                int(row["optimizer_step"])
                for row in trains
                if row["evaluation_accuracy"] is not None
            ]
            eval_spacing_ok = all(step % 32 == 0 for step in eval_steps)
            passed = bool(
                len(starts) == 1
                and len(ends) == 1
                and not failures
                and len(trains) == expected_count
                and contiguous
                and actual_interventions == expected_interventions
                and eval_spacing_ok
                and result["status"] == "completed"
                and result["termination_reason"] == "target_confirmed"
                and result["target_confirmed_step"] is not None
            )
            audit_rows.append(
                {
                    "noise_mode": noise_mode,
                    "variant": variant,
                    "records": len(records),
                    "train_steps": len(trains),
                    "contiguous": contiguous,
                    "intervention_schedule_exact": (
                        actual_interventions == expected_interventions
                    ),
                    "evaluation_spacing_exact": eval_spacing_ok,
                    "failure_records": len(failures),
                    "passed": passed,
                }
            )
            if not passed:
                raise AssertionError(f"log audit failed for {noise_mode}/{variant}")

            history = result["evaluation_history"]
            onset = sustained_onset(history, 0.10, 0.90)
            first_50 = first_step(history, 0.50)
            first_90 = first_step(history, 0.90)
            first_99 = first_step(history, 0.99)
            regularized_rows = [row for row in trains if row["regularized"]]
            ordinary_rows = [row for row in trains[5:] if not row["regularized"]]
            cosines = [
                float(row["fee_task_gradient_cosine"])
                for row in regularized_rows
                if row["fee_task_gradient_cosine"] is not None
            ]
            applied_ratios = [
                float(row["fee_applied_to_task_gradient_ratio"])
                for row in regularized_rows
                if row["fee_applied_to_task_gradient_ratio"] is not None
            ]
            regularized_seconds = [float(row["step_seconds"]) for row in regularized_rows]
            ordinary_seconds = [float(row["step_seconds"]) for row in ordinary_rows]
            intervention_median = median_or_none(regularized_seconds)
            ordinary_median = median_or_none(ordinary_seconds)
            metrics.append(
                {
                    "noise_mode": noise_mode,
                    "variant": variant,
                    "target_rate": float(result["target_intervention_rate"]),
                    "completed_steps": expected_count,
                    "confirmed_step": int(result["target_confirmed_step"]),
                    "first_50_step": first_50,
                    "first_90_step": first_90,
                    "first_99_step": first_99,
                    "sustained_10_step": onset,
                    "transition_width_sustained_10_90": (
                        first_90 - onset
                        if first_90 is not None and onset is not None
                        else None
                    ),
                    "pretransition_loss_slope_per_100_steps": (
                        loss_slope_before_transition(trains, onset)
                    ),
                    "intervention_count": len(regularized_rows),
                    "realized_rate": len(regularized_rows) / expected_count,
                    "mean_step_seconds": float(result["mean_step_seconds"]),
                    "optimization_seconds": sum(float(row["step_seconds"]) for row in trains),
                    "evaluation_seconds": sum(float(row["evaluation_seconds"]) for row in trains),
                    "ordinary_step_seconds_median": ordinary_median,
                    "intervention_step_seconds_median": intervention_median,
                    "intervention_step_slowdown": (
                        intervention_median / ordinary_median
                        if intervention_median is not None and ordinary_median is not None
                        else None
                    ),
                    "fee_task_cosine_mean": statistics.fmean(cosines) if cosines else None,
                    "fee_task_cosine_median": median_or_none(cosines),
                    "fee_task_cosine_positive_fraction": (
                        sum(value > 0 for value in cosines) / len(cosines)
                        if cosines
                        else None
                    ),
                    "applied_fee_ratio_mean": (
                        statistics.fmean(applied_ratios) if applied_ratios else None
                    ),
                    "peak_memory_bytes": int(result["peak_memory_bytes"]),
                }
            )

    metric_by_key = {(row["noise_mode"], row["variant"]): row for row in metrics}
    for noise_mode in NOISE_MODES:
        sham = metric_by_key[(noise_mode, "gs_sham")]
        horizon = sham["confirmed_step"]
        sham_seconds = sham["optimization_seconds"]
        for variant in VARIANTS:
            row = metric_by_key[(noise_mode, variant)]
            result = result_by_key[(noise_mode, variant)]
            row["confirmed_step_delta_vs_sham"] = row["confirmed_step"] - horizon
            row["confirmed_steps_saved_vs_sham"] = horizon - row["confirmed_step"]
            row["mean_step_time_ratio_vs_sham"] = (
                row["mean_step_seconds"] / sham["mean_step_seconds"]
            )
            row["optimization_time_ratio_vs_sham"] = (
                row["optimization_seconds"] / sham_seconds
            )
            row["fixed_sham_horizon_accuracy_deficit"] = fixed_horizon_accuracy_deficit(
                result["evaluation_history"], horizon, 32
            )

    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir.resolve()),
        "manifest_source_sha256": manifest["experiment_source_sha256"],
        "protocol": {
            "layers": manifest["config"]["layers"],
            "width": manifest["config"]["width"],
            "heads": manifest["config"]["heads"],
            "seed": manifest["seed"],
            "evaluation_every": manifest["config"]["evaluation_every"],
            "evaluation_batches": manifest["config"]["evaluation_batches"],
            "target_accuracy": manifest["config"]["target_token_accuracy"],
            "target_confirmations": manifest["config"]["target_confirmations"],
            "noise_start_zero_based": manifest["noise_start_zero_based"],
            "noise_duration": manifest["noise_duration"],
            "fee_gradient_ratio": manifest["fee_gradient_ratio"],
            "schedule_seed": manifest["schedule_seed"],
        },
        "audit": {
            "all_passed": all(row["passed"] for row in audit_rows),
            "trajectory_count": len(audit_rows),
            "train_step_count": sum(row["train_steps"] for row in audit_rows),
            "jsonl_record_count": sum(row["records"] for row in audit_rows),
            "rows": audit_rows,
        },
        "metrics": metrics,
    }


def _drawing_resources(width: int, height: int, scale: int = 2):
    image = Image.new("RGB", (width * scale, height * scale), "white")
    draw = ImageDraw.Draw(image, "RGBA")
    regular = "/System/Library/Fonts/Supplemental/Arial.ttf"
    bold = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
    fonts = {
        "title": ImageFont.truetype(bold, 22 * scale),
        "subtitle": ImageFont.truetype(regular, 13 * scale),
        "axis": ImageFont.truetype(regular, 12 * scale),
        "panel": ImageFont.truetype(bold, 15 * scale),
        "legend": ImageFont.truetype(regular, 12 * scale),
        "small": ImageFont.truetype(regular, 10 * scale),
    }
    return image, draw, fonts, scale


def _p(value: float, scale: int) -> int:
    return round(value * scale)


def plot_phase_curves(run_dir: Path, output: Path) -> None:
    result_summary = load_json(run_dir / "summary.json")
    by_key = {
        (row["noise_mode"], row["variant"]): row
        for row in result_summary["results"]
    }
    width, height = 1600, 1050
    image, draw, fonts, scale = _drawing_resources(width, height)
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#222}.title{font-size:22px;font-weight:600}.subtitle{font-size:13px;fill:#666}.axis{font-size:12px;fill:#555}.panel{font-size:15px;font-weight:600}.legend{font-size:12px}</style>',
        f'<text class="title" x="{width/2}" y="31" text-anchor="middle">FE-E dose response across the token-accuracy phase transition</text>',
        f'<text class="subtitle" x="{width/2}" y="55" text-anchor="middle">128 layers, width 32, seed 47; shaded interval = structured training noise; X = 99% confirmed</text>',
    ]
    draw.text(
        (_p(width / 2, scale), _p(28, scale)),
        "FE-E dose response across the token-accuracy phase transition",
        font=fonts["title"], fill="#222", anchor="mm",
    )
    draw.text(
        (_p(width / 2, scale), _p(53, scale)),
        "128 layers, width 32, seed 47; shaded interval = structured training noise; X = 99% confirmed",
        font=fonts["subtitle"], fill="#666", anchor="mm",
    )
    legend_x = 410
    for variant in VARIANTS:
        color = COLORS[variant]
        svg.extend(
            [
                f'<line x1="{legend_x}" x2="{legend_x+30}" y1="87" y2="87" stroke="{color}" stroke-width="3"/>',
                f'<circle cx="{legend_x+15}" cy="87" r="4" fill="#fff" stroke="{color}" stroke-width="2"/>',
                f'<text class="legend" x="{legend_x+39}" y="91">{escape(LABELS[variant])}</text>',
            ]
        )
        draw.line(
            (_p(legend_x, scale), _p(87, scale), _p(legend_x + 30, scale), _p(87, scale)),
            fill=color, width=3 * scale,
        )
        radius = 4 * scale
        cx, cy = _p(legend_x + 15, scale), _p(87, scale)
        draw.ellipse((cx-radius, cy-radius, cx+radius, cy+radius), fill="white", outline=color, width=2*scale)
        draw.text((_p(legend_x+39, scale), _p(87, scale)), LABELS[variant], font=fonts["legend"], fill="#333", anchor="lm")
        legend_x += 205

    panels = [(95, 140), (835, 140), (95, 595), (835, 595)]
    panel_width, panel_height = 660, 350
    for (left, top), noise_mode in zip(panels, NOISE_MODES):
        right, bottom = left + panel_width, top + panel_height
        max_confirm = max(
            int(by_key[(noise_mode, variant)]["target_confirmed_step"])
            for variant in VARIANTS
        )
        x_max = int(math.ceil((max_confirm + 64) / 200.0) * 200)

        def xmap(value: float) -> float:
            return left + value / x_max * panel_width

        def ymap(value: float) -> float:
            return bottom - value / 1.02 * panel_height

        noise_left, noise_right = xmap(128), xmap(628)
        svg.append(
            f'<rect x="{noise_left:.2f}" y="{top}" width="{noise_right-noise_left:.2f}" height="{panel_height}" fill="#9aa0a6" fill-opacity="0.14"/>'
        )
        draw.rectangle(
            (_p(noise_left, scale), _p(top, scale), _p(noise_right, scale), _p(bottom, scale)),
            fill=(154, 160, 166, 35),
        )
        for y_tick in (0.0, 0.25, 0.50, 0.75, 1.0):
            y = ymap(y_tick)
            svg.extend(
                [
                    f'<line x1="{left}" x2="{right}" y1="{y:.2f}" y2="{y:.2f}" stroke="#d5d5d5" stroke-width="1"/>',
                    f'<text class="axis" x="{left-10}" y="{y+4:.2f}" text-anchor="end">{y_tick:.2f}</text>',
                ]
            )
            draw.line((_p(left, scale), _p(y, scale), _p(right, scale), _p(y, scale)), fill="#d5d5d5", width=scale)
            draw.text((_p(left-10, scale), _p(y, scale)), f"{y_tick:.2f}", font=fonts["axis"], fill="#555", anchor="rm")
        target_y = ymap(0.99)
        for dash_start in range(left, right, 14):
            dash_end = min(dash_start + 7, right)
            svg.append(f'<line x1="{dash_start}" x2="{dash_end}" y1="{target_y:.2f}" y2="{target_y:.2f}" stroke="#777"/>')
            draw.line((_p(dash_start, scale), _p(target_y, scale), _p(dash_end, scale), _p(target_y, scale)), fill="#777", width=scale)
        svg.extend(
            [
                f'<line x1="{left}" x2="{left}" y1="{top}" y2="{bottom}" stroke="#555"/>',
                f'<line x1="{left}" x2="{right}" y1="{bottom}" y2="{bottom}" stroke="#555"/>',
                f'<text class="panel" x="{(left+right)/2}" y="{top-18}" text-anchor="middle">{escape(NOISE_LABELS[noise_mode])}</text>',
            ]
        )
        draw.line((_p(left, scale), _p(top, scale), _p(left, scale), _p(bottom, scale)), fill="#555", width=scale)
        draw.line((_p(left, scale), _p(bottom, scale), _p(right, scale), _p(bottom, scale)), fill="#555", width=scale)
        draw.text((_p((left+right)/2, scale), _p(top-18, scale)), NOISE_LABELS[noise_mode], font=fonts["panel"], fill="#222", anchor="mm")
        for x_tick in range(0, x_max + 1, 400):
            x = xmap(x_tick)
            svg.extend(
                [
                    f'<line x1="{x:.2f}" x2="{x:.2f}" y1="{bottom}" y2="{bottom+5}" stroke="#555"/>',
                    f'<text class="axis" x="{x:.2f}" y="{bottom+21}" text-anchor="middle">{x_tick}</text>',
                ]
            )
            draw.line((_p(x, scale), _p(bottom, scale), _p(x, scale), _p(bottom+5, scale)), fill="#555", width=scale)
            draw.text((_p(x, scale), _p(bottom+19, scale)), str(x_tick), font=fonts["axis"], fill="#555", anchor="mm")
        for variant in VARIANTS:
            result = by_key[(noise_mode, variant)]
            points = [
                (xmap(float(row["step"])), ymap(float(row["evaluation_accuracy"])))
                for row in result["evaluation_history"]
            ]
            path = " ".join(
                ("M" if index == 0 else "L") + f" {x:.2f} {y:.2f}"
                for index, (x, y) in enumerate(points)
            )
            color = COLORS[variant]
            svg.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2.5" stroke-linejoin="round"/>')
            scaled_points = [(_p(x, scale), _p(y, scale)) for x, y in points]
            draw.line(scaled_points, fill=color, width=3*scale, joint="curve")
            mark_every = max(1, len(points)//12)
            for index, (x, y) in enumerate(points):
                if index % mark_every:
                    continue
                svg.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3" fill="#fff" stroke="{color}" stroke-width="1.5"/>')
                radius = 3*scale
                cx, cy = _p(x, scale), _p(y, scale)
                draw.ellipse((cx-radius, cy-radius, cx+radius, cy+radius), fill="white", outline=color, width=2*scale)
            confirm_x, confirm_y = points[-1]
            svg.extend(
                [
                    f'<line x1="{confirm_x-5:.2f}" x2="{confirm_x+5:.2f}" y1="{confirm_y-5:.2f}" y2="{confirm_y+5:.2f}" stroke="{color}" stroke-width="2.5"/>',
                    f'<line x1="{confirm_x-5:.2f}" x2="{confirm_x+5:.2f}" y1="{confirm_y+5:.2f}" y2="{confirm_y-5:.2f}" stroke="{color}" stroke-width="2.5"/>',
                ]
            )
            draw.line((_p(confirm_x-5, scale), _p(confirm_y-5, scale), _p(confirm_x+5, scale), _p(confirm_y+5, scale)), fill=color, width=3*scale)
            draw.line((_p(confirm_x-5, scale), _p(confirm_y+5, scale), _p(confirm_x+5, scale), _p(confirm_y-5, scale)), fill=color, width=3*scale)
        svg.append(f'<text class="axis" x="{(left+right)/2}" y="{bottom+44}" text-anchor="middle">Optimizer step</text>')
        draw.text((_p((left+right)/2, scale), _p(bottom+42, scale)), "Optimizer step", font=fonts["axis"], fill="#555", anchor="mm")
    svg.append('<text class="axis" x="25" y="525" text-anchor="middle" transform="rotate(-90 25 525)">Token accuracy</text>')
    svg.append("</svg>")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".svg").write_text("\n".join(svg) + "\n", encoding="utf-8")
    image.save(output, dpi=(220, 220))


def plot_dose_summary(analysis: dict[str, Any], output: Path) -> None:
    metric_by_key = {
        (row["noise_mode"], row["variant"]): row for row in analysis["metrics"]
    }
    width, height = 1600, 650
    image, draw, fonts, scale = _drawing_resources(width, height)
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#222}.title{font-size:20px;font-weight:600}.axis{font-size:12px;fill:#555}.panel{font-size:15px;font-weight:600}.value{font-size:11px;font-weight:600}.legend{font-size:12px}</style>',
        f'<text class="title" x="{width/2}" y="32" text-anchor="middle">FE-E dose-response endpoint and local gradient alignment</text>',
    ]
    draw.text((_p(width/2, scale), _p(28, scale)), "FE-E dose-response endpoint and local gradient alignment", font=fonts["title"], fill="#222", anchor="mm")
    dose_variants = ("gsf_q01", "gsf_q03", "gsf_q05")
    for index, variant in enumerate(dose_variants):
        lx = 505 + index*190
        color = COLORS[variant]
        svg.extend(
            [
                f'<rect x="{lx}" y="55" width="18" height="12" fill="{color}"/>',
                f'<text class="legend" x="{lx+27}" y="66">{escape(LABELS[variant])}</text>',
            ]
        )
        draw.rectangle((_p(lx, scale), _p(55, scale), _p(lx+18, scale), _p(67, scale)), fill=color)
        draw.text((_p(lx+27, scale), _p(61, scale)), LABELS[variant], font=fonts["legend"], fill="#333", anchor="lm")

    left_box = (95, 115, 765, 540)
    l, t, r, b = left_box
    y_min, y_max = -400.0, 160.0

    def left_y(value: float) -> float:
        return b - (value-y_min)/(y_max-y_min)*(b-t)

    svg.append(f'<text class="panel" x="{(l+r)/2}" y="98" text-anchor="middle">Confirmed optimizer steps saved vs GS-SHAM (positive is better)</text>')
    draw.text((_p((l+r)/2, scale), _p(96, scale)), "Confirmed optimizer steps saved vs GS-SHAM (positive is better)", font=fonts["panel"], fill="#222", anchor="mm")
    for tick in (-400, -300, -200, -100, 0, 100):
        y = left_y(tick)
        svg.extend(
            [
                f'<line x1="{l}" x2="{r}" y1="{y:.2f}" y2="{y:.2f}" stroke="#d5d5d5"/>',
                f'<text class="axis" x="{l-10}" y="{y+4:.2f}" text-anchor="end">{tick}</text>',
            ]
        )
        draw.line((_p(l, scale), _p(y, scale), _p(r, scale), _p(y, scale)), fill="#d5d5d5", width=scale)
        draw.text((_p(l-10, scale), _p(y, scale)), str(tick), font=fonts["axis"], fill="#555", anchor="rm")
    group_width = (r-l)/4
    bar_width = 42
    for group, noise in enumerate(NOISE_MODES):
        center = l + group_width*(group+0.5)
        for offset, variant in zip((-1, 0, 1), dose_variants):
            value = metric_by_key[(noise, variant)]["confirmed_steps_saved_vs_sham"]
            x0 = center + offset*bar_width - bar_width*0.42
            x1 = center + offset*bar_width + bar_width*0.42
            y0, y1 = left_y(0), left_y(value)
            top_y, bottom_y = min(y0, y1), max(y0, y1)
            color = COLORS[variant]
            svg.append(f'<rect x="{x0:.2f}" y="{top_y:.2f}" width="{x1-x0:.2f}" height="{max(1,bottom_y-top_y):.2f}" fill="{color}"/>')
            label_y = top_y-6 if value >= 0 else bottom_y+15
            svg.append(f'<text class="value" x="{(x0+x1)/2:.2f}" y="{label_y:.2f}" text-anchor="middle">{value:+d}</text>')
            draw.rectangle((_p(x0, scale), _p(top_y, scale), _p(x1, scale), _p(max(top_y+1, bottom_y), scale)), fill=color)
            draw.text((_p((x0+x1)/2, scale), _p(label_y, scale)), f"{value:+d}", font=fonts["small"], fill="#222", anchor="mm")
        short = {"none":"None", "high_frequency":"High-freq", "energy":"Energy", "concentration":"Concentrated"}[noise]
        svg.append(f'<text class="axis" x="{center:.2f}" y="{b+25}" text-anchor="middle">{short}</text>')
        draw.text((_p(center, scale), _p(b+23, scale)), short, font=fonts["axis"], fill="#555", anchor="mm")
    zero_y = left_y(0)
    svg.append(f'<line x1="{l}" x2="{r}" y1="{zero_y:.2f}" y2="{zero_y:.2f}" stroke="#444" stroke-width="1.5"/>')
    draw.line((_p(l, scale), _p(zero_y, scale), _p(r, scale), _p(zero_y, scale)), fill="#444", width=2*scale)

    rl, rt, rr, rb = 885, 115, 1535, 540
    alignments = [
        float(metric_by_key[(noise, variant)]["fee_task_cosine_mean"])
        for noise in NOISE_MODES for variant in dose_variants
    ]
    span_min, span_max = min(alignments+[0.0]), max(alignments+[0.0])
    pad = max(0.03, (span_max-span_min)*0.18)
    align_min, align_max = span_min-pad, span_max+pad

    def right_y(value: float) -> float:
        return rb - (value-align_min)/(align_max-align_min)*(rb-rt)

    svg.append(f'<text class="panel" x="{(rl+rr)/2}" y="98" text-anchor="middle">Mean cosine(task gradient, raw FE-E gradient)</text>')
    draw.text((_p((rl+rr)/2, scale), _p(96, scale)), "Mean cosine(task gradient, raw FE-E gradient)", font=fonts["panel"], fill="#222", anchor="mm")
    tick = math.floor(align_min / 0.05) * 0.05
    while tick <= align_max + 1e-9:
        y = right_y(tick)
        svg.extend(
            [
                f'<line x1="{rl}" x2="{rr}" y1="{y:.2f}" y2="{y:.2f}" stroke="#d5d5d5"/>',
                f'<text class="axis" x="{rl-10}" y="{y+4:.2f}" text-anchor="end">{tick:.2f}</text>',
            ]
        )
        draw.line((_p(rl, scale), _p(y, scale), _p(rr, scale), _p(y, scale)), fill="#d5d5d5", width=scale)
        draw.text((_p(rl-10, scale), _p(y, scale)), f"{tick:.2f}", font=fonts["axis"], fill="#555", anchor="rm")
        tick += 0.05
    zero_y = right_y(0.0)
    svg.append(f'<line x1="{rl}" x2="{rr}" y1="{zero_y:.2f}" y2="{zero_y:.2f}" stroke="#444" stroke-width="1.5"/>')
    draw.line((_p(rl, scale), _p(zero_y, scale), _p(rr, scale), _p(zero_y, scale)), fill="#444", width=2*scale)
    slot = (rr-rl)/12
    for index, (noise, variant) in enumerate(
        (pair for noise in NOISE_MODES for pair in [(noise, v) for v in dose_variants])
    ):
        value = float(metric_by_key[(noise, variant)]["fee_task_cosine_mean"])
        center = rl + slot*(index+0.5)
        x0, x1 = center-slot*0.34, center+slot*0.34
        y_value = right_y(value)
        top_y, bottom_y = min(zero_y, y_value), max(zero_y, y_value)
        color = COLORS[variant]
        svg.append(f'<rect x="{x0:.2f}" y="{top_y:.2f}" width="{x1-x0:.2f}" height="{max(1,bottom_y-top_y):.2f}" fill="{color}"/>')
        label_y = bottom_y + 14 if value < 0 else top_y - 6
        svg.append(f'<text class="value" x="{center:.2f}" y="{label_y:.2f}" text-anchor="middle">{value:.2f}</text>')
        draw.rectangle((_p(x0, scale), _p(top_y, scale), _p(x1, scale), _p(max(top_y+1,bottom_y), scale)), fill=color)
        draw.text((_p(center, scale), _p(label_y, scale)), f"{value:.2f}", font=fonts["small"], fill="#222", anchor="mm")
        dose = int(100*metric_by_key[(noise, variant)]["target_rate"])
        svg.append(f'<text class="axis" x="{center:.2f}" y="{rb+20}" text-anchor="middle">{dose}%</text>')
        draw.text((_p(center, scale), _p(rb+18, scale)), f"{dose}%", font=fonts["small"], fill="#555", anchor="mm")
    for group, noise in enumerate(NOISE_MODES):
        center = rl + slot*(group*3+1.5)
        label = {"none":"None", "high_frequency":"High-freq", "energy":"Energy", "concentration":"Concentrated"}[noise]
        svg.append(f'<text class="axis" x="{center:.2f}" y="{rb+43}" text-anchor="middle">{label}</text>')
        draw.text((_p(center, scale), _p(rb+41, scale)), label, font=fonts["axis"], fill="#555", anchor="mm")
    svg.append("</svg>")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".svg").write_text("\n".join(svg) + "\n", encoding="utf-8")
    image.save(output, dpi=(220, 220))


def f(value: float | None, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def write_report(analysis: dict[str, Any], report_path: Path) -> None:
    by_key = {
        (row["noise_mode"], row["variant"]): row for row in analysis["metrics"]
    }
    rows = [
        "# FE-E 固定剂量—结构传播噪声相变实验报告",
        "",
        "## 结论",
        "",
        "在本次 128 层、宽度 32、seed 47 的单种子合成任务中，FE-E 不是普遍加速器。"
        "12 个 FE-E 剂量×环境配对中仅 3 个比同环境 GS-SHAM 更早确认 99%：高频噪声 1% "
        "提前 32 步，层能量集中 3% 提前 128 步、5% 提前 64 步。无噪声与全局能量放大下"
        "所有剂量均延后。最佳点是层能量集中 + 3%，确认步从 1952 降至 1824（减少 6.6%）。",
        "",
        "因此本轮支持的不是“固定频率 FE-E 应常开”，而是：**FE-E 可能在持续、局部、层深"
        "分布型失稳下存在窄剂量窗口；生产门控必须识别异常类型、介入方向与累计剂量。**",
        "",
        "## 冻结协议",
        "",
        "- 模型：128 层、宽度 32、4 头、序列长度 12、batch size 8。",
        "- 优化：AdamW + Gradient Smoothing（alpha 0.20）；seed 47；学习率 0.002。",
        "- FE-E：每次施加量固定为任务梯度范数的 5%；目标介入率 1%/3%/5%，每 100 步"
        "冻结随机抽样且 1% ⊂ 3% ⊂ 5%。",
        "- 噪声：零基第 128–627 步；验证始终为干净网络。",
        "- 验证：每 32 个优化步、8 个固定 batch。",
        "- 主终点：token accuracy ≥99% 连续 3 次；8000 步仅为安全上限。",
        "- 相变宽度：从通向首次 90% 的最后一次持续越过 10%，到首次 90%；早期偶然越过"
        "10% 后回落不计作相变起点。",
        "",
        "## 主结果",
        "",
        "| 环境 | 方法 | 确认步 | 相对 SHAM 节省步数 | 持续10%步 | 首次50% | 首次90% | 10→90宽度 | 实际介入率 | 平均步耗时/SHAM | 总优化时间/SHAM |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for noise in NOISE_MODES:
        for variant in VARIANTS:
            row = by_key[(noise, variant)]
            rows.append(
                f"| {NOISE_LABELS[noise]} | {LABELS[variant]} | {row['confirmed_step']} | "
                f"{row['confirmed_steps_saved_vs_sham']:+d} | {row['sustained_10_step']} | "
                f"{row['first_50_step']} | {row['first_90_step']} | "
                f"{row['transition_width_sustained_10_90']} | {row['realized_rate']:.2%} | "
                f"{row['mean_step_time_ratio_vs_sham']:.3f}× | "
                f"{row['optimization_time_ratio_vs_sham']:.3f}× |"
            )
    rows.extend(
        [
            "",
            "## 计算与介入方向",
            "",
            "| 环境 | 剂量 | 介入次数 | 介入步/普通步中位耗时 | 介入减速倍数 | FE-E与任务梯度平均余弦 | 正余弦比例 | 固定5%施加量审计 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for noise in NOISE_MODES:
        for variant in VARIANTS[1:]:
            row = by_key[(noise, variant)]
            rows.append(
                f"| {NOISE_LABELS[noise]} | {row['target_rate']:.0%} | "
                f"{row['intervention_count']} | {f(row['intervention_step_seconds_median'])}/"
                f"{f(row['ordinary_step_seconds_median'])} s | "
                f"{f(row['intervention_step_slowdown'], 2)}× | "
                f"{f(row['fee_task_cosine_mean'])} | "
                f"{f(row['fee_task_cosine_positive_fraction'], 2)} | "
                f"{f(row['applied_fee_ratio_mean'], 3)} |"
            )
    rows.extend(
        [
            "",
            "平均梯度余弦不能单独预测整条轨迹的收益：Adam 动量会让单次介入跨步持续，"
            "而相变由介入时点、方向和历史状态共同决定。它在这里是机制审计量，不是现成门控阈值。",
            "",
            "## 分环境解释",
            "",
            "1. **无噪声**：确认步随剂量总体变慢（1536→1600→1632→1888）。FE-E 抑制了"
            "正常训练进入相变所需的状态重组。",
            "2. **高频层间扰动**：噪声本身使 GS-SHAM 提前相变；1% FE-E 再提前 32 步，"
            "但 3%/5% 分别延后 192/128 步。异常并非越强约束越好。",
            "3. **全局能量放大**：所有 FE-E 剂量均落后，说明质量项匹配异常类型并不保证"
            "优化方向有益。",
            "4. **中层能量集中**：这是最有害的 sham 环境，也是唯一出现较强 FE-E 正收益的"
            "环境；3% 最佳、5% 次之、1% 略差，呈倒 U 型。该结果与熵约束针对层间能量集中"
            "的设计动机一致，但仍只是单种子机制证据。",
            "",
            "## 工程含义",
            "",
            "固定周期 FE-E 不适合作为生产默认策略。更合理的观测器应至少同时判断：",
            "",
            "- 异常是否持续并且是层深分布型集中，而非有益高频探索；",
            "- 原始 FE-E 梯度与任务梯度的拮抗程度是否落在可接受区间；本轮所有均值都为负，"
            "所以该量只能作为安全预算或 veto，不能把“同向”设为硬触发条件；",
            "- 最近窗口的累计介入率是否位于约 1%–3% 的候选区，而不是持续升到 5%；",
            "- 介入后是否缩短预相变平台，而非只让瞬时损失斜率更陡。",
            "",
            "生产价值应定义为“在已检测到持续传播集中时，以可控额外算力换取更早、"
            "更可靠的学习相变或避免失败”，而不是在所有训练步上平滑传播。",
            "",
            f"本机观测成本也说明必须看净收益：concentration 3% 的平均单步耗时比 sham 高 "
            f"{(by_key[('concentration', 'gsf_q03')]['mean_step_time_ratio_vs_sham']-1):.1%}，"
            f"但因少跑 128 步，总优化时间仍减少 "
            f"{(1-by_key[('concentration', 'gsf_q03')]['optimization_time_ratio_vs_sham']):.1%}；"
            f"5% 虽少跑 64 步，总优化时间反而增加 "
            f"{(by_key[('concentration', 'gsf_q05')]['optimization_time_ratio_vs_sham']-1):.1%}。"
            "所以“更新数提前”只有超过二阶介入成本后才有生产价值。",
            "",
            "## 证据边界",
            "",
            "- 单种子、微型合成反序列任务；不能外推到 7B 语言模型吞吐或最终困惑度。",
            "- 四种噪声是可控机制探针，不代表真实预训练故障的完整分布。",
            "- 多重剂量/环境筛选下的正点需要换种子复核，当前不能报告统计显著性。",
            "- 下一步优先复核 concentration 的 0%/1%/3%/5% 多种子，并将介入限制在"
            "观测到持续能量集中、且短窗口 sham/回滚探针显示净收益的窗口。",
            "",
            "## 完整性审计",
            "",
            f"- 16/16 轨迹通过；总训练步 {analysis['audit']['train_step_count']}；"
            f"JSONL 记录 {analysis['audit']['jsonl_record_count']}。",
            "- 所有日志均含唯一 run_start/run_end、连续零基步号、无 failure 记录。",
            "- 每条轨迹的实际介入步与 manifest 冻结表逐项一致；验证步均为 32 的倍数。",
            "- 16/16 在 8000 步上限前以 `target_confirmed` 结束，无删失。",
            "",
            f"原始运行目录：`{analysis['run_dir']}`",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--analysis-json",
        type=Path,
        default=Path("results/mlx_d128_s47_fee_dose_noise_acc99_analysis.json"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("docs/mlx_d128_s47_fee_dose_noise_acc99_report.md"),
    )
    parser.add_argument(
        "--phase-figure",
        type=Path,
        default=Path("output/figures/fee_d128_s47_fee_dose_noise_phase.png"),
    )
    parser.add_argument(
        "--summary-figure",
        type=Path,
        default=Path("output/figures/fee_d128_s47_fee_dose_noise_summary.png"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analysis = audit_and_analyze(args.run_dir)
    args.analysis_json.parent.mkdir(parents=True, exist_ok=True)
    args.analysis_json.write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    plot_phase_curves(args.run_dir, args.phase_figure)
    plot_dose_summary(analysis, args.summary_figure)
    write_report(analysis, args.report)
    print(f"audit: {analysis['audit']['trajectory_count']}/16 passed")
    print(f"train steps: {analysis['audit']['train_step_count']}")
    print(f"analysis: {args.analysis_json.resolve()}")
    print(f"report: {args.report.resolve()}")
    print(f"phase figure: {args.phase_figure.resolve()}")
    print(f"summary figure: {args.summary_figure.resolve()}")


if __name__ == "__main__":
    main()
