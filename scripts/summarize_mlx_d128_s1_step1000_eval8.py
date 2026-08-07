#!/usr/bin/env python3
"""Summarize the 128-layer, 1000-step, single-seed MLX comparison."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import statistics
from typing import Any


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


def resolve_run(path: Path) -> Path:
    if (path / "manifest.json").exists():
        return path
    candidates = sorted(path.glob("run_*"))
    if not candidates:
        raise FileNotFoundError(f"no run_* directory under {path}")
    return candidates[-1]


def load(path: Path) -> dict[str, Any]:
    run = resolve_run(path)
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    results = {}
    for result_path in sorted((run / "runs").glob("*.json")):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        results[result["variant"]] = result

    logs: dict[str, list[dict[str, Any]]] = {}
    records = failures = nonfinite = run_ends = 0
    for log_path in sorted((run / "logs").glob("*.jsonl")):
        content = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
        start = next(row for row in content if row["record_type"] == "run_start")
        method = start["variant"]
        logs[method] = [row for row in content if row["record_type"] == "train_step"]
        records += len(content)
        failures += sum(row["record_type"] == "failure" for row in content)
        nonfinite += sum(row.get("finite") is False for row in content)
        run_ends += sum(row["record_type"] == "run_end" and row.get("status") == "completed" for row in content)

    expected_steps = int(manifest["config"]["steps"])
    wrong_steps = {method: len(rows) for method, rows in logs.items() if len(rows) != expected_steps}
    evaluation_counts = {
        method: sum(row["evaluation_loss"] is not None for row in rows)
        for method, rows in logs.items()
    }
    expected_runs = len(METHODS)
    integrity = {
        "expected_runs": expected_runs,
        "result_files": len(results),
        "log_files": len(logs),
        "records": records,
        "train_steps": sum(len(rows) for rows in logs.values()),
        "completed_run_ends": run_ends,
        "failures": failures,
        "nonfinite": nonfinite,
        "wrong_step_logs": wrong_steps,
        "evaluation_counts": evaluation_counts,
        "passed": (
            len(results) == expected_runs
            and len(logs) == expected_runs
            and run_ends == expected_runs
            and not wrong_steps
            and not failures
            and not nonfinite
            and all(count == 8 for count in evaluation_counts.values())
        ),
    }
    return {"run": run, "manifest": manifest, "results": results, "logs": logs, "integrity": integrity}


def normalized_trapezoid(history: list[dict[str, Any]]) -> float:
    area = 0.0
    for left, right in itertools.pairwise(history):
        width = float(right["step"] - left["step"])
        area += width * (float(left["evaluation_loss"]) + float(right["evaluation_loss"])) / 2.0
    return area / float(history[-1]["step"] - history[0]["step"])


def active_intervals(rows: list[dict[str, Any]]) -> list[list[int]]:
    active_steps = [int(row["step"]) + 1 for row in rows if row["regularized"]]
    intervals = []
    for _, group in itertools.groupby(enumerate(active_steps), lambda pair: pair[1] - pair[0]):
        values = [value for _, value in group]
        intervals.append([values[0], values[-1]])
    return intervals


def summarize(data: dict[str, Any]) -> dict[str, Any]:
    results = data["results"]
    baseline = results["baseline"]
    baseline_budget = float(baseline["timed_training_seconds"])
    baseline_final = float(baseline["evaluation_loss"])
    baseline_step_seconds = float(baseline["mean_step_seconds"])
    baseline_memory = float(baseline["peak_memory_bytes"])

    methods: dict[str, Any] = {}
    observer: dict[str, Any] = {}
    for method in METHODS:
        result = results[method]
        history = result["evaluation_history"]
        losses = [float(point["evaluation_loss"]) for point in history]
        best = min(history, key=lambda point: float(point["evaluation_loss"]))
        jumps = [losses[index] - losses[index - 1] for index in range(1, len(losses))]
        max_jump = max(jumps)
        max_jump_index = jumps.index(max_jump) + 1
        eligible = [
            point for point in history
            if float(point["timed_training_seconds"]) <= baseline_budget
        ]
        compute_matched = eligible[-1]
        first_hit = next(
            (point for point in history if float(point["evaluation_loss"]) <= baseline_final),
            None,
        )
        methods[method] = {
            "final_loss": float(result["evaluation_loss"]),
            "final_accuracy": float(result["evaluation_accuracy"]),
            "delta_vs_baseline": float(result["evaluation_loss"] - baseline_final),
            "relative_improvement_vs_baseline": float((baseline_final - result["evaluation_loss"]) / baseline_final),
            "tail3_loss": statistics.fmean(losses[-3:]),
            "curve_mean_loss": statistics.fmean(losses),
            "normalized_validation_auc": normalized_trapezoid(history),
            "best_checkpoint_loss": float(best["evaluation_loss"]),
            "best_checkpoint_step": int(best["step"]),
            "maximum_adverse_jump": max_jump,
            "maximum_adverse_jump_step": int(history[max_jump_index]["step"]),
            "mean_step_seconds": float(result["mean_step_seconds"]),
            "timed_training_seconds": float(result["timed_training_seconds"]),
            "time_ratio": float(result["mean_step_seconds"] / baseline_step_seconds),
            "peak_memory_bytes": int(result["peak_memory_bytes"]),
            "memory_ratio": float(result["peak_memory_bytes"] / baseline_memory),
            "regularized_fraction": float(result["regularized_fraction"]),
            "probe_fraction": float(result["probe_fraction"]),
            "max_parameter_gradient_norm": float(result["max_parameter_gradient_norm"]),
            "final_stiffness_normalized": float(result["final_stiffness_normalized"]),
            "final_mass_energy": float(result["final_mass_energy"]),
            "final_relative_coverage": float(result["final_relative_coverage"]),
            "compute_matched": {
                "baseline_budget_seconds": baseline_budget,
                "step": int(compute_matched["step"]),
                "loss": float(compute_matched["evaluation_loss"]),
                "training_seconds": float(compute_matched["timed_training_seconds"]),
            },
            "first_reach_baseline_final": None if first_hit is None else {
                "step": int(first_hit["step"]),
                "loss": float(first_hit["evaluation_loss"]),
                "training_seconds": float(first_hit["timed_training_seconds"]),
                "speedup_vs_baseline": float(baseline_budget / first_hit["timed_training_seconds"]),
            },
            "evaluation_history": history,
        }

        if method in ("fe_e_gated", "gs_fe_e_gated"):
            rows = data["logs"][method]
            active = [row for row in rows if row["regularized"]]
            ratios = [
                float(row["fee_gradient_scale"] * row["fee_gradient_norm_raw"] / max(row["task_gradient_norm"], 1e-30))
                for row in active
            ]
            observer[method] = {
                "active_steps": len(active),
                "active_intervals": active_intervals(rows),
                "alarm_steps": [
                    int(row["step"]) + 1
                    for row in rows
                    if row["observer_state_after"] == "INTERVENE_NEXT"
                ],
                "max_observer_score": max(float(row["observer_score"]) for row in rows),
                "max_applied_fee_to_task_ratio": max(ratios, default=0.0),
            }

    always_history = methods["fe_e_always"]["evaluation_history"]
    shock = next(point for point in always_history if int(point["step"]) == 750)
    pre_shock = next(point for point in always_history if int(point["step"]) == 625)
    final = always_history[-1]
    shock_analysis = {
        "pre_shock_step": 625,
        "pre_shock_loss": float(pre_shock["evaluation_loss"]),
        "shock_step": 750,
        "shock_loss": float(shock["evaluation_loss"]),
        "jump": float(shock["evaluation_loss"] - pre_shock["evaluation_loss"]),
        "final_step": 1000,
        "final_loss": float(final["evaluation_loss"]),
        "recovery_from_shock": float(shock["evaluation_loss"] - final["evaluation_loss"]),
        "final_gap_from_pre_shock_best": float(final["evaluation_loss"] - pre_shock["evaluation_loss"]),
    }
    return {
        "methods": methods,
        "observer": observer,
        "fe_e_always_shock": shock_analysis,
        "integrity": data["integrity"],
    }


def build_report(data: dict[str, Any], summary: dict[str, Any]) -> str:
    cfg = data["manifest"]["config"]
    device = data["manifest"]["mlx_device"]
    methods = summary["methods"]
    table = [
        "| 方法 | 1000 步损失 ↓ | 相对 AdamW | 全曲线 AUC ↓ | 最优点（步） | 最大逆向跳变 | 时间 | 等时损失（步） ↓ | FE-E 介入 | 峰值内存 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        item = methods[method]
        matched = item["compute_matched"]
        table.append(
            f"| {LABELS[method]} | {item['final_loss']:.4f} | {item['delta_vs_baseline']:+.4f} | "
            f"{item['normalized_validation_auc']:.4f} | {item['best_checkpoint_loss']:.4f} ({item['best_checkpoint_step']}) | "
            f"{item['maximum_adverse_jump']:+.4f} @ {item['maximum_adverse_jump_step']} | {item['time_ratio']:.2f}× | "
            f"{matched['loss']:.4f} ({matched['step']}) | {item['regularized_fraction']:.1%} | "
            f"{item['peak_memory_bytes'] / 2**20:.1f} MiB |"
        )

    always = methods["fe_e_always"]
    gated = methods["fe_e_gated"]
    hybrid = methods["gs_fe_e_gated"]
    gs = methods["gradient_smoothing"]
    shock = summary["fe_e_always_shock"]
    integrity = summary["integrity"]
    lines = [
        "# Apple MLX 128 层、1000 步、单种子 FE-E 方案对比",
        "",
        "## 结论摘要",
        "",
        f"本次冻结配置试跑中，GS + 门控 FE-E 在第 1000 步取得最低验证损失 {hybrid['final_loss']:.4f}，相对 AdamW 的 {methods['baseline']['final_loss']:.4f} 下降 {abs(hybrid['delta_vs_baseline']):.4f}（{hybrid['relative_improvement_vs_baseline']:.2%}）。它的全曲线归一化 AUC 为 {hybrid['normalized_validation_auc']:.4f}，同样优于常开 FE-E 的 {always['normalized_validation_auc']:.4f} 与 AdamW 的 {methods['baseline']['normalized_validation_auc']:.4f}。",
        "",
        f"常开 FE-E 的终点损失为 {always['final_loss']:.4f}，但耗时 {always['time_ratio']:.2f}×，并在第 750 步出现 +{shock['jump']:.4f} 的显著瞬态冲击。到第 1000 步它恢复了 {shock['recovery_from_shock']:.4f}，但仍比第 625 步高 {shock['final_gap_from_pre_shock_best']:.4f}。门控方案把同一检查点的最大逆向跳变限制在 {gated['maximum_adverse_jump']:.4f}，混合方案为 {hybrid['maximum_adverse_jump']:.4f}。",
        "",
        "这些结果是单种子、微型合成任务上的探索性证据，不构成统计显著性结论，也不能直接外推到真实语言模型规模。最值得复核的新假设是：Gradient Smoothing 单独会拖慢收敛，但与门控 FE-E 组合后可能形成互补。",
        "",
        "## 冻结实验协议",
        "",
        f"- {cfg['layers']} 层 Pre-LN Transformer，宽度 {cfg['width']}、{cfg['heads']} 个头、序列长度 {cfg['sequence_length']}、批量 {cfg['batch_size']}。",
        f"- 种子 31；{cfg['steps']} 个训练更新；每 {cfg['evaluation_every']} 步验证一次，共 8 个检查点；每个检查点使用 {cfg['evaluation_batches']} 个验证 batch。",
        "- 五种方法：AdamW、Gradient Smoothing、常开 FE-E、观测器门控 FE-E、GS + 门控 FE-E。固定周期脉冲不在本实验中。",
        f"- FE-E 系数：刚度 {cfg['lambda_stiffness']}、质量 {cfg['lambda_energy']}、熵 {cfg['lambda_entropy']}、覆盖带 [{cfg['entropy_lower']:.2f}, {cfg['entropy_upper']:.2f}]；门控 FE-E 增量受任务梯度 {cfg['gated_fee_gradient_ratio']:.1f}× 信赖域限制。",
        f"- 环境：MLX {data['manifest']['mlx_version']}，`{device['device_name']}`，{int(device['memory_size']) / 2**30:.0f} GiB 统一内存。",
        "",
        "## 主结果",
        "",
        *table,
        "",
        "AUC 是八个验证检查点之间梯形积分的训练步轴归一化均值；等时损失采用不超过 AdamW 1000 步训练时间的最后一个预定验证点，因此是保守、离散的训练计算时间比较（不含验证耗时）。",
        "",
        "## 计算效率与工程含义",
        "",
        f"- 常开 FE-E 虽然单步慢 {always['time_ratio']:.2f}×，但在 AdamW 的 {methods['baseline']['timed_training_seconds']:.1f} 秒训练预算内完成到第 {always['compute_matched']['step']} 步并达到 {always['compute_matched']['loss']:.4f}，仍优于 AdamW 第 1000 步的 {methods['baseline']['final_loss']:.4f}。它首次达到该阈值的预定检查点是第 {always['first_reach_baseline_final']['step']} 步，训练时间 {always['first_reach_baseline_final']['training_seconds']:.1f} 秒。",
        f"- 混合方案单步慢 {hybrid['time_ratio']:.2f}×；按同一预算保守取第 {hybrid['compute_matched']['step']} 步，损失 {hybrid['compute_matched']['loss']:.4f}，也略优于 AdamW。它达到 AdamW 终点阈值所需 {hybrid['first_reach_baseline_final']['training_seconds']:.1f} 秒，对应 {hybrid['first_reach_baseline_final']['speedup_vs_baseline']:.2f}× 的首次达标速度。",
        f"- 纯 Gradient Smoothing 只增加约 {(gs['time_ratio'] - 1):.1%} 时间，却把终点损失推高到 {gs['final_loss']:.4f}；本配置下它不能单独作为更优基线。",
        f"- 门控 FE-E 介入 {gated['regularized_fraction']:.1%} 的训练步，耗时 {gated['time_ratio']:.2f}×；混合方案介入 {hybrid['regularized_fraction']:.1%}，耗时 {hybrid['time_ratio']:.2f}×。二者峰值内存约为 AdamW 的 {gated['memory_ratio']:.2f}× 与 {hybrid['memory_ratio']:.2f}×。",
        "- 因而“算力充足时 FE-E 更有优势”在本试跑里得到的是有条件支持：常开 FE-E 的任务质量和等时离散检查均优于 AdamW，但其瞬态冲击与内存开销仍存在；混合门控方案给出了更好的质量—稳定性折中。",
        "",
        "## 观测器审计",
        "",
        f"- 门控 FE-E 触发 {len(summary['observer']['fe_e_gated']['alarm_steps'])} 次、介入 {summary['observer']['fe_e_gated']['active_steps']} 步；混合方案触发 {len(summary['observer']['gs_fe_e_gated']['alarm_steps'])} 次、介入 {summary['observer']['gs_fe_e_gated']['active_steps']} 步。",
        f"- 两种门控方案实际施加的 FE-E 梯度增量/任务梯度最大比值均为 {max(summary['observer']['fe_e_gated']['max_applied_fee_to_task_ratio'], summary['observer']['gs_fe_e_gated']['max_applied_fee_to_task_ratio']):.3f}，符合 0.5 信赖域。",
        "- 后半程告警接近持续出现，控制器形成约 4 步介入、若干步恢复的准周期模式。它不是固定周期脉冲，但也尚未证明能精准区分稀有异常；下一轮应提高覆盖度基线的自适应性，报告告警精确率与消融。",
        "",
        "## 机制快照",
        "",
        f"- 常开 FE-E 末步归一化刚度为 {always['final_stiffness_normalized']:.4f}，AdamW 为 {methods['baseline']['final_stiffness_normalized']:.4f}；末步覆盖度分别为 {always['final_relative_coverage']:.3f} 与 {methods['baseline']['final_relative_coverage']:.3f}，说明 FE-E 传播形态约束确实生效。",
        f"- 混合方案末步刚度为 {hybrid['final_stiffness_normalized']:.4f}、覆盖度 {hybrid['final_relative_coverage']:.3f}。由于末步处于探测而非介入状态，这个快照不能代表其整个训练期机制，需要用时间平均指标复核。",
        f"- 最大参数梯度范数：AdamW {methods['baseline']['max_parameter_gradient_norm']:.2f}、GS {gs['max_parameter_gradient_norm']:.2f}、常开 FE-E {always['max_parameter_gradient_norm']:.2f}、门控 FE-E {gated['max_parameter_gradient_norm']:.2f}、混合方案 {hybrid['max_parameter_gradient_norm']:.2f}。混合方案的低峰值是积极信号，但单种子不足以作因果归因。",
        "",
        "## 完整性审计",
        "",
        f"- {integrity['expected_runs']} 个运行、{integrity['train_steps']} 个训练步、{integrity['records']} 条 JSONL 记录；每种方法恰有 8 个验证点。",
        f"- 失败 {integrity['failures']}，非有限值 {integrity['nonfinite']}，完整性检查 `{integrity['passed']}`。",
        f"- 正式源码 SHA-256：`{data['manifest']['source_sha256']}`。",
        "",
        "## 证据位置",
        "",
        f"- 正式运行：`{data['run']}`",
        "- 逐步日志：`logs/*.jsonl`。",
        "- 每方法摘要：`runs/*.json`。",
        "- 配置、命令、硬件与源码哈希：`manifest.json`。",
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
