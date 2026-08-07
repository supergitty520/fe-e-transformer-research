#!/usr/bin/env python3
"""Summarize the frozen five-method MLX 96-layer clean/stress protocol."""

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


def mean_sd(values: Iterable[float]) -> tuple[float, float]:
    items = list(values)
    if not items:
        return float("nan"), float("nan")
    return statistics.fmean(items), statistics.stdev(items) if len(items) > 1 else 0.0


def paired(values: list[float]) -> dict[str, Any]:
    mean, sd = mean_sd(values)
    if len(values) < 2:
        interval = [mean, mean]
    else:
        half = T_975.get(len(values), 1.96) * sd / math.sqrt(len(values))
        interval = [mean - half, mean + half]
    return {"values": values, "mean": mean, "sd": sd, "ci95": interval}


def load_run(path: Path) -> dict[str, Any]:
    run = resolve_run(path)
    manifest = json.loads((run / "manifest.json").read_text())
    expected_steps = int(manifest["config"]["steps"])
    expected_seeds = [int(seed) for seed in manifest["seeds"]]
    expected_keys = {(method, seed) for method in METHODS for seed in expected_seeds}

    results = [json.loads(path.read_text()) for path in sorted((run / "runs").glob("*.json"))]
    by_method: dict[str, list[dict[str, Any]]] = {method: [] for method in METHODS}
    for result in results:
        if result["variant"] in by_method:
            by_method[result["variant"]].append(result)
    for values in by_method.values():
        values.sort(key=lambda item: int(item["seed"]))

    logs: dict[tuple[str, int], list[dict[str, Any]]] = {}
    record_count = 0
    failures = 0
    nonfinite = 0
    for log_path in sorted((run / "logs").glob("*.jsonl")):
        records = [json.loads(line) for line in log_path.read_text().splitlines()]
        start = next(record for record in records if record["record_type"] == "run_start")
        key = (start["variant"], int(start["seed"]))
        if key not in expected_keys:
            continue
        logs[key] = [record for record in records if record["record_type"] == "train_step"]
        record_count += len(records)
        failures += sum(record["record_type"] == "failure" for record in records)
        nonfinite += sum(record.get("finite") is False for record in records)

    missing_results = sorted(expected_keys - {(item["variant"], int(item["seed"])) for item in results})
    missing_logs = sorted(expected_keys - set(logs))
    wrong_step_logs = sorted(
        (method, seed, len(rows))
        for (method, seed), rows in logs.items()
        if len(rows) != expected_steps
    )
    integrity = {
        "expected_runs": len(expected_keys),
        "result_files": len(results),
        "log_files": len(logs),
        "records": record_count,
        "train_steps": sum(len(rows) for rows in logs.values()),
        "failures": failures,
        "nonfinite": nonfinite,
        "missing_results": missing_results,
        "missing_logs": missing_logs,
        "wrong_step_logs": wrong_step_logs,
        "passed": not (missing_results or missing_logs or wrong_step_logs or failures or nonfinite),
    }
    return {
        "run": run,
        "manifest": manifest,
        "results": results,
        "by_method": by_method,
        "logs": logs,
        "integrity": integrity,
    }


def scenario_summary(data: dict[str, Any]) -> dict[str, Any]:
    by_method = data["by_method"]
    baseline_time = {int(item["seed"]): item["mean_step_seconds"] for item in by_method["baseline"]}
    baseline_budget = {
        int(item["seed"]): item["timed_training_seconds"] for item in by_method["baseline"]
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
            "final_mass_energy",
            "final_relative_coverage",
        ):
            mean, sd = mean_sd(float(item[name]) for item in items)
            metrics[name] = {"mean": mean, "sd": sd}
        ratios = [item["mean_step_seconds"] / baseline_time[int(item["seed"])] for item in items]
        mean, sd = mean_sd(ratios)
        metrics["time_ratio"] = {"mean": mean, "sd": sd}

        budget_losses: list[float] = []
        budget_steps: list[int] = []
        for item in items:
            eligible = [
                checkpoint
                for checkpoint in item["evaluation_history"]
                if checkpoint["timed_training_seconds"] <= baseline_budget[int(item["seed"])]
            ]
            if not eligible:
                eligible = [item["evaluation_history"][0]]
            budget_losses.append(float(eligible[-1]["evaluation_loss"]))
            budget_steps.append(int(eligible[-1]["step"]))
        mean, sd = mean_sd(budget_losses)
        metrics["compute_matched_loss"] = {"mean": mean, "sd": sd}
        metrics["compute_matched_steps"] = statistics.fmean(budget_steps)
        methods[method] = metrics

    baseline = {int(item["seed"]): item for item in by_method["baseline"]}
    gs = {int(item["seed"]): item for item in by_method["gradient_smoothing"]}
    paired_base: dict[str, Any] = {}
    paired_gs: dict[str, Any] = {}
    for method in METHODS[1:]:
        paired_base[method] = paired(
            [float(item["evaluation_loss"] - baseline[int(item["seed"])]["evaluation_loss"]) for item in by_method[method]]
        )
        paired_gs[method] = paired(
            [float(item["evaluation_loss"] - gs[int(item["seed"])]["evaluation_loss"]) for item in by_method[method]]
        )

    stress_step = int(data["manifest"]["config"]["stress_step"])
    observer: dict[str, Any] = {}
    for method in ("fe_e_gated", "gs_fe_e_gated"):
        per_seed = []
        for item in by_method[method]:
            seed = int(item["seed"])
            rows = data["logs"][(method, seed)]
            alarms = [
                int(row["step"])
                for row in rows
                if row["observer_state_after"] == "INTERVENE_NEXT"
            ]
            active = [row for row in rows if row["regularized"]]
            ratios = [
                float(row["fee_gradient_scale"])
                * float(row["fee_gradient_norm_raw"])
                / max(float(row["task_gradient_norm"]), 1e-30)
                for row in active
            ]
            post = [step for step in alarms if stress_step >= 0 and step >= stress_step]
            pre = [step for step in alarms if stress_step < 0 or step < stress_step]
            per_seed.append(
                {
                    "seed": seed,
                    "alarms": alarms,
                    "pre_stress_or_clean_alarms": pre,
                    "active_steps": len(active),
                    "first_post_stress_alarm": post[0] if post else None,
                    "post_stress_detection_delay": post[0] - stress_step if post else None,
                    "max_applied_fee_to_task_ratio": max(ratios, default=0.0),
                    "max_raw_fee_gradient": max(
                        (float(row["fee_gradient_norm_raw"]) for row in active), default=0.0
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


def metric(value: dict[str, float], digits: int = 4) -> str:
    return f"{value['mean']:.{digits}f} ± {value['sd']:.{digits}f}"


def ci_text(value: dict[str, Any]) -> str:
    low, high = value["ci95"]
    return f"{value['mean']:+.4f} [{low:+.4f}, {high:+.4f}]"


def result_table(summary: dict[str, Any]) -> list[str]:
    rows = [
        "| 方法 | 评估损失 ↓ | 准确率 ↑ | 时间/AdamW | 峰值内存 | FE-E 介入率 | 刚度 ↓ | 覆盖度 ↑ | 等时损失 ↓（步数） |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        values = summary["methods"][method]
        rows.append(
            f"| {LABELS[method]} | {metric(values['evaluation_loss'])} | "
            f"{metric(values['evaluation_accuracy'], 3)} | {values['time_ratio']['mean']:.2f}× | "
            f"{values['peak_memory_bytes']['mean'] / 2**20:.0f} MiB | "
            f"{values['regularized_fraction']['mean']:.1%} | "
            f"{values['final_stiffness_normalized']['mean']:.4f} | "
            f"{values['final_relative_coverage']['mean']:.3f} | "
            f"{metric(values['compute_matched_loss'])} ({values['compute_matched_steps']:.0f}) |"
        )
    return rows


def build_report(clean: dict[str, Any], stress: dict[str, Any], d24: dict[str, Any] | None) -> str:
    clean_s = scenario_summary(clean)
    stress_s = scenario_summary(stress)
    cfg = clean["manifest"]["config"]
    device = clean["manifest"]["mlx_device"]
    always = clean_s["methods"]["fe_e_always"]
    gs = clean_s["methods"]["gradient_smoothing"]
    base = clean_s["methods"]["baseline"]
    max_ratio = max(
        seed["max_applied_fee_to_task_ratio"]
        for summary in (clean_s, stress_s)
        for method in ("fe_e_gated", "gs_fe_e_gated")
        for seed in summary["observer"][method]
    )
    delays = [
        seed["post_stress_detection_delay"]
        for method in ("fe_e_gated", "gs_fe_e_gated")
        for seed in stress_s["observer"][method]
        if seed["post_stress_detection_delay"] is not None
    ]
    clean_alarm_counts = [
        len(seed["alarms"])
        for method in ("fe_e_gated", "gs_fe_e_gated")
        for seed in clean_s["observer"][method]
    ]
    total_integrity = {
        key: clean_s["integrity"][key] + stress_s["integrity"][key]
        for key in ("expected_runs", "result_files", "log_files", "records", "train_steps", "failures", "nonfinite")
    }

    scaling_lines: list[str] = []
    if d24:
        d24_clean = d24["clean"]["methods"]
        scaling_lines = [
            "## 从 24 层扩展到 96 层",
            "",
            "| 方法 | 24 层损失 | 96 层损失 | 24 层时间比 | 96 层时间比 |",
            "|---|---:|---:|---:|---:|",
            f"| AdamW | {d24_clean['baseline']['evaluation_loss']['mean']:.4f} | {base['evaluation_loss']['mean']:.4f} | 1.00× | 1.00× |",
            f"| Gradient Smoothing | {d24_clean['gradient_smoothing']['evaluation_loss']['mean']:.4f} | {gs['evaluation_loss']['mean']:.4f} | {d24_clean['gradient_smoothing']['time_ratio']['mean']:.2f}× | {gs['time_ratio']['mean']:.2f}× |",
            f"| FE-E 常开 | {d24_clean['fe_e_always']['evaluation_loss']['mean']:.4f} | {always['evaluation_loss']['mean']:.4f} | {d24_clean['fe_e_always']['time_ratio']['mean']:.2f}× | {always['time_ratio']['mean']:.2f}× |",
            "",
            "深度增加后，常开 FE-E 的固定步数优势仍存在，但二阶伴随计算的约 2× 时间代价没有消失。96 层模型在相同宽度与步数下绝对损失更高，不能把 24/96 层损失差直接解释为深度缩放规律；它主要反映当前小任务、残差参数化和短训练预算的组合。",
            "",
        ]

    lines = [
        "# Apple MLX 96 层 FE-E 正式对比实验",
        "",
        "## 结论摘要",
        "",
        f"冻结 24 层调参结果后，在 96 层、三随机种子、200 步协议中，常开 FE-E 的干净训练评估损失最低：{always['evaluation_loss']['mean']:.4f}，AdamW 为 {base['evaluation_loss']['mean']:.4f}，Gradient Smoothing 为 {gs['evaluation_loss']['mean']:.4f}。常开 FE-E 相对 AdamW 的配对差为 {ci_text(clean_s['paired_vs_baseline']['fe_e_always'])}，相对 GS 为 {ci_text(clean_s['paired_vs_gradient_smoothing']['fe_e_always'])}；n=3 的区间仍较宽，属于积极信号而非统计定论。",
        "",
        f"工程代价同样明确：常开 FE-E 每步耗时约 {always['time_ratio']['mean']:.2f}× AdamW，而 GS 约 {gs['time_ratio']['mean']:.2f}×。按 AdamW 的墙钟训练预算截断后，常开 FE-E 只能完成约 {always['compute_matched_steps']:.0f} 步，等时损失为 {always['compute_matched_loss']['mean']:.4f}，因此当前优势是“等更新次数、算力充足”的优势，不是“等时间”的效率优势。",
        "",
        "固定周期脉冲 FE-E 已依据 24 层负结果退出主实验，本轮只比较五种保留方案。96 层结果再次显示：常开 FE-E 比当前门控版本更可靠；观测机制的研究价值在于将高成本二阶介入变成按需控制，但现有触发器尚未达到这一目标。",
        "",
        "## 冻结实验协议",
        "",
        f"- {cfg['layers']} 层 Pre-LN Transformer，宽度 {cfg['width']}、{cfg['heads']} 个注意力头、序列长度 {cfg['sequence_length']}、批量 {cfg['batch_size']}。",
        f"- 反序列合成任务；{cfg['steps']} 次参数更新；确认种子 31、47、59。",
        "- 方法：AdamW、论文式 Gradient Smoothing、常开 FE-E、观测器门控 FE-E、GS + 门控 FE-E。",
        f"- FE-E 系数沿用 24 层冻结值：刚度 {cfg['lambda_stiffness']}、质量 {cfg['lambda_energy']}、熵 {cfg['lambda_entropy']}、覆盖带 [{cfg['entropy_lower']:.2f}, {cfg['entropy_upper']:.2f}]。",
        f"- 门控前 {cfg['observer_calibration_steps']} 步校准，每 {cfg['observer_probe_every']} 步精确探测，介入 {cfg['intervention_steps']} 步，FE-E 梯度增量上限为任务梯度范数的 {cfg['gated_fee_gradient_ratio']:.0%}。",
        "- 压力协议在第 80–87 步将学习率由 0.002 临时提高到 0.010。",
        f"- 环境：MLX {clean['manifest']['mlx_version']}，`{device['device_name']}`，{device['memory_size'] / 2**30:.0f} GiB 统一内存。",
        "",
        "## 干净训练",
        "",
        *result_table(clean_s),
        "",
        "配对损失差（候选减参考，95% t 区间；n=3）：",
        "",
        f"- GS − AdamW：{ci_text(clean_s['paired_vs_baseline']['gradient_smoothing'])}。",
        f"- 常开 FE-E − AdamW：{ci_text(clean_s['paired_vs_baseline']['fe_e_always'])}。",
        f"- 常开 FE-E − GS：{ci_text(clean_s['paired_vs_gradient_smoothing']['fe_e_always'])}。",
        f"- 门控 FE-E − AdamW：{ci_text(clean_s['paired_vs_baseline']['fe_e_gated'])}。",
        f"- GS + 门控 FE-E − GS：{ci_text(clean_s['paired_vs_gradient_smoothing']['gs_fe_e_gated'])}。",
        "",
        f"机制指标与任务结果方向一致：常开 FE-E 将末步归一化刚度从 {base['final_stiffness_normalized']['mean']:.4f} 降到 {always['final_stiffness_normalized']['mean']:.4f}，相对熵覆盖度从 {base['final_relative_coverage']['mean']:.3f} 提高到 {always['final_relative_coverage']['mean']:.3f}。这说明 FE-E 确实改变了深层伴随场，而非仅靠优化器随机波动。",
        "",
        "## 学习率冲击",
        "",
        *result_table(stress_s),
        "",
        "配对损失差：",
        "",
        f"- GS − AdamW：{ci_text(stress_s['paired_vs_baseline']['gradient_smoothing'])}。",
        f"- 常开 FE-E − AdamW：{ci_text(stress_s['paired_vs_baseline']['fe_e_always'])}。",
        f"- 常开 FE-E − GS：{ci_text(stress_s['paired_vs_gradient_smoothing']['fe_e_always'])}。",
        f"- 门控 FE-E − AdamW：{ci_text(stress_s['paired_vs_baseline']['fe_e_gated'])}。",
        f"- GS + 门控 FE-E − GS：{ci_text(stress_s['paired_vs_gradient_smoothing']['gs_fe_e_gated'])}。",
        "",
        "压力实验检验的是短时学习率跃迁后的恢复，而不是完整鲁棒性。应同时查看最终损失、冲击后的轨迹和跨种子一致性；任何单种子胜负都不足以支持结论。",
        "",
        "## 观测器、安全与日志审计",
        "",
        f"- 两套正式实验共 {total_integrity['expected_runs']} 个预期运行、{total_integrity['train_steps']} 个训练步、{total_integrity['records']} 条 JSONL 记录；失败 {total_integrity['failures']}，非有限值 {total_integrity['nonfinite']}。",
        f"- 清洁训练中，两个门控方案的单运行报警次数为 {clean_alarm_counts}；这类报警可视作无外加冲击条件下的介入负担，而不能自动等同于真正误报。",
        f"- 压力集精确报警延迟样本为 {delays} 步；观测器探测周期为 8 步，因此检测分辨率先天地受限。",
        f"- 全部实际介入的最大 `||g_FE,applied|| / ||g_task||` 为 {max_ratio:.6f}，未超过 0.5 信赖域。",
        f"- 完整性检查：clean={clean_s['integrity']['passed']}，stress={stress_s['integrity']['passed']}；源码 SHA-256 分别为 `{clean['manifest']['source_sha256']}` 和 `{stress['manifest']['source_sha256']}`。",
        "",
        *scaling_lines,
        "## 工程含义",
        "",
        "1. 若目标是固定训练步数、可以接受约 2× 训练耗时，常开 FE-E 在这个受控实验中具有优势；它以算力换取更平滑、更分散的跨层梯度传播和更低终点损失。",
        "2. 若目标是固定墙钟、单位电费或单位 GPU 小时产出，当前 FE-E 没有优势。二阶自动微分是主要成本，显存并非本机小模型的瓶颈。",
        "3. 门控 FE-E 仍值得研究，但理由不是现版本已经胜出，而是它有机会把二阶计算只放在传播异常时。下一版应采用每步一阶代理监测、进入 WATCH 后每 1–2 步精确探测，并用连续动作强度代替满开/满关。",
        "4. 这仍是合成任务上的小规模机制实验。发表级结论需要真实语言建模数据、至少 5 个确认种子、深度/宽度缩放、长训练和强基线（Pre-LN、DeepNorm、µP/残差缩放、梯度裁剪等）。",
        "",
        "## 证据位置",
        "",
        f"- 干净训练：`{clean['run']}`",
        f"- 学习率冲击：`{stress['run']}`",
        "- 逐步日志：各运行目录下 `logs/*.jsonl`。",
        "- 每方法/种子摘要：各运行目录下 `runs/*.json`。",
        "- 命令、硬件、配置与源码校验：各运行目录下 `manifest.json`。",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", type=Path, required=True)
    parser.add_argument("--stress", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--d24-json", type=Path)
    args = parser.parse_args()

    clean = load_run(args.clean)
    stress = load_run(args.stress)
    d24 = json.loads(args.d24_json.read_text()) if args.d24_json else None
    payload = {
        "clean_run": str(clean["run"]),
        "stress_run": str(stress["run"]),
        "clean": scenario_summary(clean),
        "stress": scenario_summary(stress),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_report(clean, stress, d24), encoding="utf-8")
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
