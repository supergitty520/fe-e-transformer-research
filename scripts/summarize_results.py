#!/usr/bin/env python3
"""Aggregate replicated JSON runs without third-party dependencies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics


DISPLAY_NAMES = {"fe_entropy": "FE-E"}


def mean_sd(values: list[float], digits: int = 4) -> str:
    mean = statistics.fmean(values)
    if len(values) == 1:
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {statistics.stdev(values):.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--comparison-step", type=int, default=35)
    args = parser.parse_args()

    payloads = [json.loads(path.read_text()) for path in args.inputs]
    variants = [item["variant"] for item in payloads[0]["results"]]
    by_payload = [
        {item["variant"]: item for item in payload["results"]}
        for payload in payloads
    ]
    baseline_times = [
        result_map["baseline"]["mean_step_seconds"] for result_map in by_payload
    ]

    lines = [
        "# 三随机种子消融汇总",
        "",
        f"共同诊断步：{args.comparison_step}；数值为均值 ± 样本标准差。",
        "",
        "| 变体 | 评估损失 ↓ | 准确率 ↑ | 相对刚度 ↓ | 质量能量 | 深度覆盖度 | 每步耗时/基线 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    machine_rows: list[dict[str, object]] = []
    for variant in variants:
        results = [result_map[variant] for result_map in by_payload]
        diagnostics = [result["history"][args.comparison_step] for result in results]
        eval_loss = [float(result["evaluation_loss"]) for result in results]
        eval_accuracy = [float(result["evaluation_accuracy"]) for result in results]
        stiffness = [float(item["stiffness_normalized"]) for item in diagnostics]
        mass = [float(item["mass_energy"]) for item in diagnostics]
        coverage = [float(item["relative_coverage"]) for item in diagnostics]
        time_ratio = [
            float(result["mean_step_seconds"]) / baseline
            for result, baseline in zip(results, baseline_times, strict=True)
        ]
        lines.append(
            "| "
            + " | ".join(
                [
                    DISPLAY_NAMES.get(variant, variant),
                    mean_sd(eval_loss),
                    mean_sd(eval_accuracy, 3),
                    mean_sd(stiffness),
                    mean_sd(mass, 8),
                    mean_sd(coverage, 3),
                    mean_sd(time_ratio, 2) + "×",
                ]
            )
            + " |"
        )
        machine_rows.append(
            {
                "variant": variant,
                "evaluation_loss": eval_loss,
                "evaluation_accuracy": eval_accuracy,
                "stiffness_normalized": stiffness,
                "mass_energy": mass,
                "relative_coverage": coverage,
                "step_time_ratio": time_ratio,
            }
        )

    lines.extend(
        [
            "",
            "说明：基线每 5 步做一次梯度诊断，而正则变体每步做二阶反传，因此耗时比是当前实现上的保守下界，不是大模型吞吐结论。",
            "",
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines))
    machine_output = args.output.with_suffix(".json")
    machine_output.write_text(
        json.dumps(
            {
                "inputs": [str(path) for path in args.inputs],
                "comparison_step": args.comparison_step,
                "rows": machine_rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
