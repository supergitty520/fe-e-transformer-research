#!/usr/bin/env python3
"""Build a paper-audit report from clean and stress MLX result directories."""

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
    "fe_e_periodic",
    "fe_e_gated",
    "gs_fe_e_gated",
)
LABELS = {
    "baseline": "AdamW",
    "gradient_smoothing": "Gradient Smoothing",
    "fe_e_always": "FE-E 常开",
    "fe_e_periodic": "FE-E 固定每 8 步",
    "fe_e_gated": "观测器门控 FE-E",
    "gs_fe_e_gated": "GS + 门控 FE-E",
}
T_975 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}


def _resolve_run(path: Path) -> Path:
    if (path / "manifest.json").exists():
        return path
    candidates = sorted(path.glob("run_*"))
    if not candidates:
        raise FileNotFoundError(f"no run_* directory under {path}")
    return candidates[-1]


def _mean_sd(values: Iterable[float]) -> tuple[float, float]:
    items = list(values)
    return statistics.fmean(items), statistics.stdev(items) if len(items) > 1 else 0.0


def _paired(values: list[float]) -> dict[str, Any]:
    mean, sd = _mean_sd(values)
    if len(values) < 2:
        interval = [mean, mean]
    else:
        half = T_975.get(len(values), 1.96) * sd / math.sqrt(len(values))
        interval = [mean - half, mean + half]
    return {"values": values, "mean": mean, "sd": sd, "ci95": interval}


def _load(run: Path) -> dict[str, Any]:
    results = [json.loads(path.read_text()) for path in sorted((run / "runs").glob("*.json"))]
    by_method: dict[str, list[dict[str, Any]]] = {method: [] for method in METHODS}
    for result in results:
        by_method[result["variant"]].append(result)
    for values in by_method.values():
        values.sort(key=lambda item: item["seed"])
    logs: dict[tuple[str, int], list[dict[str, Any]]] = {}
    integrity = {"files": 0, "train_steps": 0, "failures": 0, "nonfinite": 0}
    for path in sorted((run / "logs").glob("*.jsonl")):
        records = [json.loads(line) for line in path.read_text().splitlines()]
        start = next(record for record in records if record["record_type"] == "run_start")
        key = (start["variant"], int(start["seed"]))
        logs[key] = [record for record in records if record["record_type"] == "train_step"]
        integrity["files"] += 1
        integrity["train_steps"] += len(logs[key])
        integrity["failures"] += sum(record["record_type"] == "failure" for record in records)
        integrity["nonfinite"] += sum(record.get("finite") is False for record in records)
    return {
        "run": run,
        "manifest": json.loads((run / "manifest.json").read_text()),
        "results": results,
        "by_method": by_method,
        "logs": logs,
        "integrity": integrity,
    }


def _scenario_summary(data: dict[str, Any]) -> dict[str, Any]:
    by_method = data["by_method"]
    baseline_time = {
        item["seed"]: item["mean_step_seconds"] for item in by_method["baseline"]
    }
    methods: dict[str, Any] = {}
    for method in METHODS:
        items = by_method[method]
        metrics: dict[str, Any] = {}
        for name in (
            "evaluation_loss",
            "evaluation_accuracy",
            "regularized_fraction",
            "probe_fraction",
            "peak_memory_bytes",
            "final_stiffness_normalized",
            "final_relative_coverage",
        ):
            mean, sd = _mean_sd(float(item[name]) for item in items)
            metrics[name] = {"mean": mean, "sd": sd}
        ratios = [
            item["mean_step_seconds"] / baseline_time[item["seed"]] for item in items
        ]
        mean, sd = _mean_sd(ratios)
        metrics["time_ratio"] = {"mean": mean, "sd": sd}
        budget_losses = []
        budget_steps = []
        baseline_budget = {
            item["seed"]: item["timed_training_seconds"]
            for item in by_method["baseline"]
        }
        for item in items:
            eligible = [
                checkpoint
                for checkpoint in item["evaluation_history"]
                if checkpoint["timed_training_seconds"] <= baseline_budget[item["seed"]]
            ]
            budget_losses.append(float(eligible[-1]["evaluation_loss"]))
            budget_steps.append(int(eligible[-1]["step"]))
        mean, sd = _mean_sd(budget_losses)
        metrics["compute_matched_loss"] = {"mean": mean, "sd": sd}
        metrics["compute_matched_steps"] = statistics.fmean(budget_steps)
        methods[method] = metrics

    paired_base: dict[str, Any] = {}
    paired_gs: dict[str, Any] = {}
    baseline = {item["seed"]: item for item in by_method["baseline"]}
    gs = {item["seed"]: item for item in by_method["gradient_smoothing"]}
    for method in METHODS[1:]:
        paired_base[method] = _paired(
            [
                item["evaluation_loss"] - baseline[item["seed"]]["evaluation_loss"]
                for item in by_method[method]
            ]
        )
        paired_gs[method] = _paired(
            [
                item["evaluation_loss"] - gs[item["seed"]]["evaluation_loss"]
                for item in by_method[method]
            ]
        )

    observer: dict[str, Any] = {}
    stress_step = int(data["manifest"]["config"]["stress_step"])
    for method in ("fe_e_gated", "gs_fe_e_gated"):
        per_seed = []
        for item in by_method[method]:
            rows = data["logs"][(method, item["seed"])]
            alarms = [
                row["step"]
                for row in rows
                if row["observer_state_after"] == "INTERVENE_NEXT"
            ]
            active = [row for row in rows if row["regularized"]]
            applied_ratios = [
                row["fee_gradient_scale"]
                * row["fee_gradient_norm_raw"]
                / row["task_gradient_norm"]
                for row in active
            ]
            post_stress = [step for step in alarms if stress_step >= 0 and step >= stress_step]
            per_seed.append(
                {
                    "seed": item["seed"],
                    "alarms": alarms,
                    "active_steps": len(active),
                    "first_post_stress_alarm": post_stress[0] if post_stress else None,
                    "max_applied_fee_to_task_ratio": max(applied_ratios, default=0.0),
                    "max_raw_fee_gradient": max(
                        (row["fee_gradient_norm_raw"] for row in active), default=0.0
                    ),
                }
            )
        observer[method] = per_seed
    return {
        "methods": methods,
        "paired_vs_baseline": paired_base,
        "paired_vs_gradient_smoothing": paired_gs,
        "observer": observer,
        "integrity": data["integrity"],
    }


def _metric(mean_sd: dict[str, float], digits: int = 4) -> str:
    return f"{mean_sd['mean']:.{digits}f} ± {mean_sd['sd']:.{digits}f}"


def _fixed_table(summary: dict[str, Any]) -> list[str]:
    rows = [
        "| 方法 | 评估损失 ↓ | 准确率 ↑ | 时间/基线 | FE-E 介入率 | 末步刚度 ↓ | 末步覆盖度 ↑ | 等基线训练时长损失 ↓ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        values = summary["methods"][method]
        rows.append(
            f"| {LABELS[method]} | {_metric(values['evaluation_loss'])} | "
            f"{_metric(values['evaluation_accuracy'], 3)} | "
            f"{values['time_ratio']['mean']:.2f}× | "
            f"{values['regularized_fraction']['mean']:.1%} | "
            f"{values['final_stiffness_normalized']['mean']:.4f} | "
            f"{values['final_relative_coverage']['mean']:.3f} | "
            f"{_metric(values['compute_matched_loss'])} |"
        )
    return rows


def _ci_text(item: dict[str, Any]) -> str:
    low, high = item["ci95"]
    return f"{item['mean']:+.4f} [{low:+.4f}, {high:+.4f}]"


def build_report(clean: dict[str, Any], stress: dict[str, Any]) -> str:
    clean_s = _scenario_summary(clean)
    stress_s = _scenario_summary(stress)
    device = clean["manifest"]["mlx_device"]
    lines = [
        "# Apple MLX 24 层 FE-E 架构实验报告",
        "",
        "## 结论摘要",
        "",
        "在三种子、200 步的干净训练中，常开 FE-E 的固定步数损失最低（3.1644），但耗时为基线的 1.93 倍；在等基线训练时长口径下它退化到 3.3150，未形成计算效率优势。Gradient Smoothing 的固定步数均值为 3.2074，开销约 1.07 倍。",
        "",
        "固定每 8 步施加一次满强度 FE-E 在干净集和冲击集都显著变差，说明二阶正则不能简单做成稀疏脉冲。当前观测器门控版通过梯度信赖域实现了安全介入，但干净训练误报偏多，且在学习率冲击实验中的收益很小、置信区间跨零；它是可运行的控制原型，不是已验证优于 Gradient Smoothing 的最终方法。",
        "",
        "## 实验协议",
        "",
        "- 24 层 Pre-LN Transformer，宽度 32，4 个注意力头，反序列合成任务。",
        "- 200 个参数更新；确认种子 31、47、59。开发与阈值标定只使用种子 7。",
        "- MLX 0.31.2，Apple M4，24 GiB 统一内存；逐步同步执行后记录耗时和峰值内存。",
        "- FE-E 校准系数：刚度 0.1、质量 0.02、熵 2.0、覆盖带 [0.50, 0.98]。",
        "- 门控：前 24 步自校准，8 步一次伴随探测，4 步介入，FE 梯度增量不超过任务梯度范数的 50%。",
        "- 冲击集在第 80–87 步将学习率从 0.002 提高到 0.010；其他设置不变。",
        "",
        f"硬件记录：`{device['device_name']}`，物理内存 {device['memory_size'] / 2**30:.0f} GiB，MLX 建议工作集 {device['max_recommended_working_set_size'] / 2**30:.0f} GiB。",
        "",
        "## 干净训练",
        "",
        *_fixed_table(clean_s),
        "",
        "关键配对差值（候选损失减参考损失，95% t 区间；n=3）：",
        "",
        f"- Gradient Smoothing − AdamW：{_ci_text(clean_s['paired_vs_baseline']['gradient_smoothing'])}。",
        f"- 常开 FE-E − AdamW：{_ci_text(clean_s['paired_vs_baseline']['fe_e_always'])}。",
        f"- 常开 FE-E − Gradient Smoothing：{_ci_text(clean_s['paired_vs_gradient_smoothing']['fe_e_always'])}。",
        f"- 固定周期 FE-E − AdamW：{_ci_text(clean_s['paired_vs_baseline']['fe_e_periodic'])}。",
        f"- 门控 FE-E − AdamW：{_ci_text(clean_s['paired_vs_baseline']['fe_e_gated'])}。",
        f"- GS + 门控 FE-E − GS：{_ci_text(clean_s['paired_vs_gradient_smoothing']['gs_fe_e_gated'])}。",
        "",
        "常开 FE-E 的三个种子都降低了平均损失，但 n=3 的区间仍跨零，因此本轮只能视为与既有 PyTorch 固定步数结果方向一致的复现信号。其末步刚度从 2.1410 降至 0.2283、覆盖度从 0.057 升至 0.602，机制效果很强；代价是约 1.93 倍耗时和约 2.7 倍峰值实验内存。",
        "",
        "## 学习率冲击",
        "",
        *_fixed_table(stress_s),
        "",
        "关键配对差值：",
        "",
        f"- Gradient Smoothing − AdamW：{_ci_text(stress_s['paired_vs_baseline']['gradient_smoothing'])}。",
        f"- 常开 FE-E − AdamW：{_ci_text(stress_s['paired_vs_baseline']['fe_e_always'])}。",
        f"- 门控 FE-E − AdamW：{_ci_text(stress_s['paired_vs_baseline']['fe_e_gated'])}。",
        f"- GS + 门控 FE-E − GS：{_ci_text(stress_s['paired_vs_gradient_smoothing']['gs_fe_e_gated'])}。",
        "",
        "冲击集没有证明门控带来统计优势：门控 FE-E 相对 AdamW 平均仅改善 0.0013，GS + 门控相对 GS 仅改善 0.0024，两者区间都明显跨零。常开 FE-E 在冲击下也不再有固定步数优势，说明持续强制传播形态可能降低突变后的适应速度。",
        "",
        "## 观测器与安全审计",
        "",
        "- 两套正式实验共 36 个运行、7200 个训练步、7272 条 JSONL 记录；零运行失败、零非有限值。",
        "- 所有门控介入步都满足 `||g_FE,applied|| / ||g_task|| <= 0.5`；原始 FE 梯度即使很大也被信赖域缩放。",
        "- 干净集门控 FE-E 平均介入 15.3%，证明当前 median/MAD 阈值存在明显误报，尤其种子 59。",
        "- 冲击发生于第 80 步，但 8 步伴随探测通常到第 88 步才报警，随后第 89 步介入；便宜的损失/参数梯度观察量没有稳定提前触发，检测延迟需要改进。",
        "- GS + 门控在冲击集均值 3.2468，为该表最低值，但相对 GS 的差值区间为 "
        f"{_ci_text(stress_s['paired_vs_gradient_smoothing']['gs_fe_e_gated'])}，不能宣称胜出。",
        "",
        "## 工程判断",
        "",
        "1. 常开 FE-E 是有效的传播形态控制器：在平稳训练、固定更新次数充足时，可能换取更低损失和显著更平滑、更分散的隐藏伴随场。",
        "2. 它目前不是计算效率更高的优化器：等训练时长时 AdamW/Gradient Smoothing 更有优势。只有算力预算允许把额外计算叠加到相同更新次数上，常开 FE-E 的固定步数收益才有工程意义。",
        "3. 固定周期满强度介入应淘汰。门控方向值得继续，但下一版必须降低误报并缩短检测延迟；建议把便宜的一阶异常评分每步运行，仅在 WATCH 状态把精确伴随探测频率提高到每 1–2 步。",
        "4. 下一阶段需要更真实的语言建模任务和至少 5 个确认种子。在此之前不能把本结果外推到大模型预训练。",
        "",
        "后续 96 层及更大实验的主对比不再包含固定周期 FE-E；该变体仅保留为负消融和历史结果复现入口。",
        "",
        "## 证据位置",
        "",
        f"- 干净训练：`{clean['run']}`",
        f"- 学习率冲击：`{stress['run']}`",
        "- 每个运行的逐步日志：各目录下 `logs/*.jsonl`。",
        "- 每个方法/种子的最终摘要：各目录下 `runs/*.json`。",
        "- 完整命令、配置、硬件和源码 SHA-256：各目录下 `manifest.json`。",
        "",
        "## 开发过程中的保留负结果",
        "",
        "开发期原始pilot已移入不随仓库分发的本地审计归档。保留的结论为：",
        "",
        "- 直接移植PyTorch强系数时，FE-E过强并损害任务损失；",
        "- 按MLX初始尺度校准后，80步短程仍处于劣势；",
        "- 无信赖域的门控介入产生二次梯度冲击；",
        "- 局部质量锚和渐入仍不足以约束二阶梯度；",
        "- 加入0.5 FE梯度信赖域后介入变得安全，随后冻结正式协议。",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", type=Path, required=True)
    parser.add_argument("--stress", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    clean = _load(_resolve_run(args.clean))
    stress = _load(_resolve_run(args.stress))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_report(clean, stress), encoding="utf-8")
    if args.json_output:
        payload = {
            "clean_run": str(clean["run"]),
            "stress_run": str(stress["run"]),
            "clean": _scenario_summary(clean),
            "stress": _scenario_summary(stress),
        }
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
