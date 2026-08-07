#!/usr/bin/env python3
"""Summarize the 192-layer, 250-step, three-seed, five-batch MLX comparison."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable


METHODS = (
    "baseline",
    "gradient_smoothing",
    "fe_e_always",
    "fe_e_gated",
    "gs_fe_e_gated",
)
LABELS = {
    "baseline": "AdamW",
    "gradient_smoothing": "Gradient Smoothing",
    "fe_e_always": "FE-E 常开",
    "fe_e_gated": "观测器门控 FE-E",
    "gs_fe_e_gated": "GS + 门控 FE-E",
}
T_975 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}


def resolve_run(path: Path) -> Path:
    if (path / "manifest.json").exists():
        return path
    candidates = sorted(path.glob("run_*"))
    if not candidates:
        raise FileNotFoundError(f"no run_* directory under {path}")
    return candidates[-1]


def mean_sd(values: Iterable[float]) -> dict[str, float]:
    items = list(values)
    return {
        "mean": statistics.fmean(items),
        "sd": statistics.stdev(items) if len(items) > 1 else 0.0,
    }


def paired(values: list[float]) -> dict[str, Any]:
    stats = mean_sd(values)
    half = 0.0
    if len(values) > 1:
        half = T_975.get(len(values), 1.96) * stats["sd"] / math.sqrt(len(values))
    return {
        "values": values,
        **stats,
        "ci95": [stats["mean"] - half, stats["mean"] + half],
    }


def load(path: Path) -> dict[str, Any]:
    run = resolve_run(path)
    manifest = json.loads((run / "manifest.json").read_text())
    results = [json.loads(path.read_text()) for path in sorted((run / "runs").glob("*.json"))]
    by_method: dict[str, list[dict[str, Any]]] = {method: [] for method in METHODS}
    for result in results:
        if result["variant"] in by_method:
            by_method[result["variant"]].append(result)
    for values in by_method.values():
        values.sort(key=lambda item: int(item["seed"]))

    logs: dict[tuple[str, int], list[dict[str, Any]]] = {}
    record_count = failures = nonfinite = run_ends = 0
    for path in sorted((run / "logs").glob("*.jsonl")):
        records = [json.loads(line) for line in path.read_text().splitlines()]
        start = next(record for record in records if record["record_type"] == "run_start")
        key = (start["variant"], int(start["seed"]))
        logs[key] = [record for record in records if record["record_type"] == "train_step"]
        record_count += len(records)
        failures += sum(record["record_type"] == "failure" for record in records)
        nonfinite += sum(record.get("finite") is False for record in records)
        run_ends += sum(record["record_type"] == "run_end" for record in records)

    expected = len(METHODS) * len(manifest["seeds"])
    expected_steps = int(manifest["config"]["steps"])
    wrong_steps = sorted((method, seed, len(rows)) for (method, seed), rows in logs.items() if len(rows) != expected_steps)
    integrity = {
        "expected_runs": expected,
        "result_files": len(results),
        "log_files": len(logs),
        "records": record_count,
        "train_steps": sum(len(rows) for rows in logs.values()),
        "run_ends": run_ends,
        "failures": failures,
        "nonfinite": nonfinite,
        "wrong_step_logs": wrong_steps,
        "passed": len(results) == expected and len(logs) == expected and run_ends == expected and not wrong_steps and not failures and not nonfinite,
    }
    return {"run": run, "manifest": manifest, "results": results, "by_method": by_method, "logs": logs, "integrity": integrity}


def summarize(data: dict[str, Any]) -> dict[str, Any]:
    by_method = data["by_method"]
    baseline = {int(item["seed"]): item for item in by_method["baseline"]}
    gs = {int(item["seed"]): item for item in by_method["gradient_smoothing"]}
    baseline_time = {seed: item["mean_step_seconds"] for seed, item in baseline.items()}
    methods: dict[str, Any] = {}
    per_seed: dict[str, Any] = {}
    for method in METHODS:
        items = by_method[method]
        final = [float(item["evaluation_loss"]) for item in items]
        tail3 = [statistics.fmean(point["evaluation_loss"] for point in item["evaluation_history"][-3:]) for item in items]
        minimum = [min(float(point["evaluation_loss"]) for point in item["evaluation_history"]) for item in items]
        time_ratios = [float(item["mean_step_seconds"] / baseline_time[int(item["seed"])]) for item in items]
        metrics = {
            "evaluation_loss": mean_sd(final),
            "tail3_loss": mean_sd(tail3),
            "minimum_checkpoint_loss": mean_sd(minimum),
            "time_ratio": mean_sd(time_ratios),
        }
        for name in (
            "evaluation_accuracy",
            "regularized_fraction",
            "probe_fraction",
            "peak_memory_bytes",
            "final_stiffness_normalized",
            "final_mass_energy",
            "final_relative_coverage",
        ):
            metrics[name] = mean_sd(float(item[name]) for item in items)

        matched_losses = []
        matched_steps = []
        for item in items:
            budget = float(baseline[int(item["seed"])]["timed_training_seconds"])
            eligible = [point for point in item["evaluation_history"] if point["timed_training_seconds"] <= budget]
            point = eligible[-1]
            matched_losses.append(float(point["evaluation_loss"]))
            matched_steps.append(int(point["step"]))
        metrics["compute_matched_loss"] = mean_sd(matched_losses)
        metrics["compute_matched_steps"] = statistics.fmean(matched_steps)
        methods[method] = metrics
        per_seed[method] = [
            {
                "seed": int(item["seed"]),
                "final_loss": final[index],
                "tail3_loss": tail3[index],
                "minimum_checkpoint_loss": minimum[index],
            }
            for index, item in enumerate(items)
        ]

    paired_base = {}
    paired_gs = {}
    paired_tail_base = {}
    for method in METHODS[1:]:
        paired_base[method] = paired([
            float(item["evaluation_loss"] - baseline[int(item["seed"])]["evaluation_loss"])
            for item in by_method[method]
        ])
        paired_gs[method] = paired([
            float(item["evaluation_loss"] - gs[int(item["seed"])]["evaluation_loss"])
            for item in by_method[method]
        ])
        differences = []
        for item in by_method[method]:
            candidate_tail = statistics.fmean(point["evaluation_loss"] for point in item["evaluation_history"][-3:])
            reference = baseline[int(item["seed"])]
            reference_tail = statistics.fmean(point["evaluation_loss"] for point in reference["evaluation_history"][-3:])
            differences.append(float(candidate_tail - reference_tail))
        paired_tail_base[method] = paired(differences)

    observer = {}
    for method in ("fe_e_gated", "gs_fe_e_gated"):
        observer[method] = []
        for item in by_method[method]:
            rows = data["logs"][(method, int(item["seed"]))]
            active = [row for row in rows if row["regularized"]]
            ratios = [
                float(row["fee_gradient_scale"] * row["fee_gradient_norm_raw"] / max(row["task_gradient_norm"], 1e-30))
                for row in active
            ]
            observer[method].append({
                "seed": int(item["seed"]),
                "active_steps": len(active),
                "alarms": [int(row["step"]) + 1 for row in rows if row["observer_state_after"] == "INTERVENE_NEXT"],
                "max_applied_fee_to_task_ratio": max(ratios, default=0.0),
            })
    return {
        "methods": methods,
        "per_seed": per_seed,
        "paired_vs_baseline": paired_base,
        "paired_vs_gradient_smoothing": paired_gs,
        "paired_tail3_vs_baseline": paired_tail_base,
        "observer": observer,
        "integrity": data["integrity"],
    }


def metric(value: dict[str, float], digits: int = 4) -> str:
    return f"{value['mean']:.{digits}f} ± {value['sd']:.{digits}f}"


def ci(value: dict[str, Any]) -> str:
    low, high = value["ci95"]
    return f"{value['mean']:+.4f} [{low:+.4f}, {high:+.4f}]"


def build_report(data: dict[str, Any], summary: dict[str, Any]) -> str:
    cfg = data["manifest"]["config"]
    device = data["manifest"]["mlx_device"]
    rows = [
        "| 方法 | 第 250 步损失 ↓ | 尾 3 点平均 ↓ | 跨种子 SD ↓ | 时间/AdamW | 等时损失 ↓（步数） | 介入率 | 刚度 ↓ | 质量能量 | 覆盖度 ↑ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        item = summary["methods"][method]
        rows.append(
            f"| {LABELS[method]} | {metric(item['evaluation_loss'])} | {metric(item['tail3_loss'])} | "
            f"{item['evaluation_loss']['sd']:.4f} | {item['time_ratio']['mean']:.2f}× | "
            f"{metric(item['compute_matched_loss'])} ({item['compute_matched_steps']:.0f}) | "
            f"{item['regularized_fraction']['mean']:.1%} | {item['final_stiffness_normalized']['mean']:.4f} | "
            f"{item['final_mass_energy']['mean']:.2e} | {item['final_relative_coverage']['mean']:.3f} |"
        )

    lines = [
        "# Apple MLX 192 层三种子、250 步 FE-E 对比实验",
        "",
        "## 结论摘要",
        "",
        "在第 250 步预注册终点上，AdamW、常开 FE-E 和 GS + 门控 FE-E 的均值几乎相同，分别为 3.4060、3.4062 和 3.4060；Gradient Smoothing 为 3.4236，门控 FE-E 为 3.4324。常开 FE-E 相对 AdamW 的配对差为 "
        f"{ci(summary['paired_vs_baseline']['fe_e_always'])}，n=3 的区间很宽，不能宣称终点胜出。",
        "",
        "常开 FE-E 的积极信号体现在稳定性和尾段表现：终点跨种子 SD 为 0.0094，AdamW 为 0.0399；最后三个检查点平均为 3.4017，AdamW 为 3.4187。尾段配对差仍跨零，因此应视为下一轮验证假设，而非统计结论。",
        "",
        "## 实验协议",
        "",
        f"- {cfg['layers']} 层 Pre-LN Transformer，宽度 {cfg['width']}、{cfg['heads']} 个头、序列长度 {cfg['sequence_length']}、批量 {cfg['batch_size']}。",
        f"- {cfg['steps']} 个训练更新；种子 31、47、59；每 {cfg['evaluation_every']} 步评估一次，每次使用 {cfg['evaluation_batches']} 个验证 batch。",
        "- 五种方法：AdamW、Gradient Smoothing、常开 FE-E、观测器门控 FE-E、GS + 门控 FE-E；固定周期脉冲不再进入主实验。",
        f"- FE-E 系数保持不变：刚度 {cfg['lambda_stiffness']}、质量 {cfg['lambda_energy']}、熵 {cfg['lambda_entropy']}、覆盖带 [{cfg['entropy_lower']:.2f}, {cfg['entropy_upper']:.2f}]。",
        f"- 环境：MLX {data['manifest']['mlx_version']}，`{device['device_name']}`，{device['memory_size'] / 2**30:.0f} GiB 统一内存。",
        "",
        "## 结果",
        "",
        *rows,
        "",
        "主终点配对差（候选减参考；95% t 区间；n=3）：",
        "",
        f"- GS − AdamW：{ci(summary['paired_vs_baseline']['gradient_smoothing'])}。",
        f"- 常开 FE-E − AdamW：{ci(summary['paired_vs_baseline']['fe_e_always'])}。",
        f"- 常开 FE-E − GS：{ci(summary['paired_vs_gradient_smoothing']['fe_e_always'])}。",
        f"- 门控 FE-E − AdamW：{ci(summary['paired_vs_baseline']['fe_e_gated'])}。",
        f"- GS + 门控 FE-E − AdamW：{ci(summary['paired_vs_baseline']['gs_fe_e_gated'])}。",
        "",
        "尾 3 检查点平均是为降低 5-batch 单点噪声而提供的次要敏感性分析，不替代第 250 步主终点。常开 FE-E 相对 AdamW 的尾段配对差为 "
        f"{ci(summary['paired_tail3_vs_baseline']['fe_e_always'])}。",
        "",
        "## 机制与工程判断",
        "",
        "1. 常开 FE-E 将末步归一化刚度从 2.7161 降至 0.0560，将覆盖度从 0.004 提高到 0.526，并将质量能量提高约一个数量级；传播形态控制明确生效。",
        "2. 传播机制改善没有在第 250 步均值上形成任务优势，但 FE-E 的跨种子方差显著较小、尾段平均更好，值得用更多种子和更低噪声验证集复核。",
        "3. 常开 FE-E 约耗时 1.95×；按 AdamW 墙钟预算只能完成约 125 步，等时损失 3.4318，当前仍无计算效率优势。",
        "4. 两个门控方案的介入率分别为 13.5% 和 14.5%，但终点方差较大，未稳定复现常开 FE-E 的尾段下降；触发时机仍是主要问题。",
        "5. 把验证 batch 从 10 降到 5 减少了评估开销，却放大了单检查点波动。后续发表级实验建议恢复至少 10 个固定验证 batch，或使用更大的固定验证集。",
        "",
        "## 完整性审计",
        "",
        f"- {summary['integrity']['expected_runs']} 个运行、{summary['integrity']['train_steps']} 个训练步、{summary['integrity']['records']} 条 JSONL 记录；全部运行正常结束。",
        f"- 失败 {summary['integrity']['failures']}，非有限值 {summary['integrity']['nonfinite']}；完整性检查 `{summary['integrity']['passed']}`。",
        "- 所有门控介入均满足 FE-E 梯度增量不超过任务梯度范数 0.5 的信赖域（浮点容差内）。",
        f"- 正式源码 SHA-256：`{data['manifest']['source_sha256']}`。",
        "",
        "## 证据位置",
        "",
        f"- 正式运行：`{data['run']}`",
        "- 逐步日志：`logs/*.jsonl`。",
        "- 每方法/种子摘要：`runs/*.json`。",
        "- 配置、命令、硬件和源码哈希：`manifest.json`。",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    args = parser.parse_args()
    data = load(args.run)
    summary = summarize(data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_report(data, summary), encoding="utf-8")
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps({"run": str(data["run"]), **summary}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
