#!/usr/bin/env python3
"""Summarize paired FE-E vs Gradient Smoothing confirmation runs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics


T_CRITICAL_95_DF4 = 2.776
VARIANTS = ("baseline", "gradient_smoothing", "fe_entropy")
DISPLAY_NAMES = {
    "baseline": "AdamW",
    "gradient_smoothing": "Gradient Smoothing",
    "fe_entropy": "FE-E",
}


def mean_sd(values: list[float], digits: int = 4) -> str:
    return (
        f"{statistics.fmean(values):.{digits}f} ± "
        f"{statistics.stdev(values):.{digits}f}"
    )


def paired_ci(values: list[float]) -> tuple[float, float, float]:
    mean = statistics.fmean(values)
    half_width = T_CRITICAL_95_DF4 * statistics.stdev(values) / math.sqrt(len(values))
    return mean, mean - half_width, mean + half_width


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diagnostic-step", type=int, default=180)
    parser.add_argument("--fe-compute-step", type=int, default=100)
    args = parser.parse_args()
    if len(args.inputs) != 5:
        raise ValueError("this paired summary expects five confirmation seeds")

    payloads = [json.loads(path.read_text()) for path in args.inputs]
    runs = [
        {result["variant"]: result for result in payload["results"]}
        for payload in payloads
    ]
    for run in runs:
        if set(run) != set(VARIANTS):
            raise ValueError("each input must contain baseline, gradient_smoothing, fe_entropy")

    rows: dict[str, dict[str, list[float]]] = {}
    for variant in VARIANTS:
        rows[variant] = {
            "evaluation_loss": [run[variant]["evaluation_loss"] for run in runs],
            "evaluation_accuracy": [
                run[variant]["evaluation_accuracy"] for run in runs
            ],
            "time_ratio": [
                run[variant]["mean_step_seconds"]
                / run["baseline"]["mean_step_seconds"]
                for run in runs
            ],
            "stiffness": [
                run[variant]["history"][args.diagnostic_step][
                    "stiffness_normalized"
                ]
                for run in runs
            ],
            "mass": [
                run[variant]["history"][args.diagnostic_step]["mass_energy"]
                for run in runs
            ],
            "coverage": [
                run[variant]["history"][args.diagnostic_step]["relative_coverage"]
                for run in runs
            ],
            "residual_cosine": [
                run[variant]["history"][args.diagnostic_step][
                    "residual_adjacent_cosine"
                ]
                for run in runs
            ],
            "residual_lss": [
                run[variant]["history"][args.diagnostic_step][
                    "residual_line_shape_score"
                ]
                for run in runs
            ],
            "update_raw_roughness": [
                run[variant]["history"][args.diagnostic_step][
                    "update_raw_roughness"
                ]
                for run in runs
            ],
            "update_applied_roughness": [
                run[variant]["history"][args.diagnostic_step][
                    "update_applied_roughness"
                ]
                for run in runs
            ],
            "update_applied_cosine": [
                run[variant]["history"][args.diagnostic_step][
                    "update_applied_adjacent_cosine"
                ]
                for run in runs
            ],
        }

    paired: dict[str, dict[str, tuple[float, float, float]]] = {}
    for variant in ("gradient_smoothing", "fe_entropy"):
        paired[variant] = {}
        for metric in ("evaluation_loss", "evaluation_accuracy"):
            differences = [
                candidate - baseline
                for candidate, baseline in zip(
                    rows[variant][metric], rows["baseline"][metric], strict=True
                )
            ]
            paired[variant][metric] = paired_ci(differences)
    paired["fe_vs_gs"] = {}
    for metric in ("evaluation_loss", "evaluation_accuracy"):
        differences = [
            fe - smoothing
            for fe, smoothing in zip(
                rows["fe_entropy"][metric],
                rows["gradient_smoothing"][metric],
                strict=True,
            )
        ]
        paired["fe_vs_gs"][metric] = paired_ci(differences)

    # A generous approximate equal-backward budget: FE at 100 steps versus the
    # first-order methods at 200, although measured FE cost is usually >2x.
    compute_budget: dict[str, list[float]] = {}
    for variant in VARIANTS:
        values: list[float] = []
        for run in runs:
            if variant == "fe_entropy":
                checkpoint = next(
                    item
                    for item in run[variant]["evaluation_history"]
                    if item["step"] == args.fe_compute_step
                )
                values.append(checkpoint["evaluation_loss"])
            else:
                values.append(run[variant]["evaluation_loss"])
        compute_budget[variant] = values

    # Actual wall-clock budget: each seed uses its baseline's total elapsed time.
    wall_clock: dict[str, dict[str, list[float] | list[int]]] = {}
    for variant in VARIANTS:
        losses: list[float] = []
        steps: list[int] = []
        for run in runs:
            budget = run["baseline"]["elapsed_seconds"]
            eligible = [
                item
                for item in run[variant]["evaluation_history"]
                if item["elapsed_seconds"] <= budget
            ]
            checkpoint = eligible[-1] if eligible else run[variant]["evaluation_history"][0]
            losses.append(checkpoint["evaluation_loss"])
            steps.append(checkpoint["step"])
        wall_clock[variant] = {"loss": losses, "step": steps}

    lines = [
        "# FE-E 与 Gradient Smoothing 直接对照",
        "",
        "确认集：5 个未参与超参数选择的随机种子；24 层、宽度 32；200 个更新步。",
        "Gradient Smoothing 为论文 Window Standard/Proj，`alpha=0.2`；FE-E 使用上一阶段固定系数。",
        "",
        "## 固定更新步数",
        "",
        "| 方法 | 评估损失 ↓ | 准确率 ↑ | 每步时间/基线 | 伴随刚度 ↓ | 质量能量 | 覆盖度 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        row = rows[variant]
        lines.append(
            "| "
            + " | ".join(
                [
                    DISPLAY_NAMES[variant],
                    mean_sd(row["evaluation_loss"]),
                    mean_sd(row["evaluation_accuracy"], 3),
                    mean_sd(row["time_ratio"], 2) + "×",
                    mean_sd(row["stiffness"]),
                    mean_sd(row["mass"], 8),
                    mean_sd(row["coverage"], 3),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "配对差值（候选−基线）的 95% t 区间：",
            "",
            f"- Gradient Smoothing 损失：{paired['gradient_smoothing']['evaluation_loss'][0]:.4f} "
            f"[{paired['gradient_smoothing']['evaluation_loss'][1]:.4f}, {paired['gradient_smoothing']['evaluation_loss'][2]:.4f}]。",
            f"- FE-E 损失：{paired['fe_entropy']['evaluation_loss'][0]:.4f} "
            f"[{paired['fe_entropy']['evaluation_loss'][1]:.4f}, {paired['fe_entropy']['evaluation_loss'][2]:.4f}]。",
            f"- FE-E 相对 Gradient Smoothing 损失：{paired['fe_vs_gs']['evaluation_loss'][0]:.4f} "
            f"[{paired['fe_vs_gs']['evaluation_loss'][1]:.4f}, {paired['fe_vs_gs']['evaluation_loss'][2]:.4f}]。",
            "",
            "## 计算预算口径",
            "",
            "| 方法 | 约等反传预算损失 ↓ | 基线墙钟预算损失 ↓ | 墙钟预算平均步数 |",
            "|---|---:|---:|---:|",
        ]
    )
    for variant in VARIANTS:
        lines.append(
            f"| {DISPLAY_NAMES[variant]} | {mean_sd(compute_budget[variant])} | "
            f"{mean_sd(wall_clock[variant]['loss'])} | "
            f"{statistics.fmean(wall_clock[variant]['step']):.1f} |"
        )

    raw = rows["gradient_smoothing"]["update_raw_roughness"]
    applied = rows["gradient_smoothing"]["update_applied_roughness"]
    reduction = [1.0 - after / before for before, after in zip(raw, applied, strict=True)]
    lines.extend(
        [
            "",
            "## 机制指标",
            "",
            f"- Gradient Smoothing 将投影层 AdamW 更新粗糙度平均降低 {statistics.fmean(reduction) * 100:.1f}%。",
            f"- 应用更新的相邻余弦：基线 {statistics.fmean(rows['baseline']['update_applied_cosine']):.3f}，"
            f"Gradient Smoothing {statistics.fmean(rows['gradient_smoothing']['update_applied_cosine']):.3f}。",
            f"- 隐藏伴随刚度：Gradient Smoothing {statistics.fmean(rows['gradient_smoothing']['stiffness']):.4f}，"
            f"FE-E {statistics.fmean(rows['fe_entropy']['stiffness']):.4f}。",
            f"- 表示残差相邻余弦：基线 {statistics.fmean(rows['baseline']['residual_cosine']):.3f}，"
            f"Gradient Smoothing {statistics.fmean(rows['gradient_smoothing']['residual_cosine']):.3f}，"
            f"FE-E {statistics.fmean(rows['fe_entropy']['residual_cosine']):.3f}。本实验没有复现论文在 ViT 上的表示残差对齐提升。",
            "",
            "约等反传预算把 FE-E 的 100 步与一阶方法的 200 步比较；这实际上略偏向 FE-E，因为其观测成本通常超过 2 倍。墙钟结果受共享 CPU 负载影响，应结合种子内时间比解释。",
            "",
        ]
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines))
    args.output.with_suffix(".json").write_text(
        json.dumps(
            {
                "inputs": [str(path) for path in args.inputs],
                "rows": rows,
                "paired_95_ci": paired,
                "compute_budget": compute_budget,
                "wall_clock": wall_clock,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
