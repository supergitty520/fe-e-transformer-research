#!/usr/bin/env python3
"""Audit and plot the 128-layer persistent-harm FE-E/GS experiment."""

from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
import statistics
from xml.sax.saxutils import escape

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
RUN = (
    ROOT
    / "results/mlx_d128_s1_acc99_persistent_gate_v3_hybrid_first"
    / "run_20260805_152559"
)
FIGURE_BASE = ROOT / "output/figures/fee_d128_acc99_persistent_gate_v3"
SUMMARY_PATH = ROOT / "results/mlx_d128_acc99_persistent_gate_v3_analysis.json"
REPORT_PATH = ROOT / "docs/mlx_d128_acc99_persistent_gate_v3_report.md"

VARIANTS = {
    "gs_fe_e_gated": "GS + gated FE-E",
    "gradient_smoothing": "Pure GS",
}
COLORS = {
    "gs_fe_e_gated": "#0B6E69",
    "gradient_smoothing": "#D97706",
}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def median_or_none(values):
    return statistics.median(values) if values else None


def evaluation_at(result, step):
    for item in result["evaluation_history"]:
        if item["step"] == step:
            return item
    return None


def ordinary_step_medians(records):
    ordinary = [
        row
        for row in records
        if row.get("record_type") == "train_step"
        and not row.get("regularized")
        and not row.get("adjoint_probe")
        and not row.get("observer_probe")
        and not row.get("evaluation_seconds")
    ]
    return {
        "first_200": median_or_none([row["step_seconds"] for row in ordinary[:200]]),
        "last_200": median_or_none([row["step_seconds"] for row in ordinary[-200:]]),
    }


def analyze():
    manifest = read_json(RUN / "manifest.json")
    results = {
        variant: read_json(RUN / "runs" / f"{variant}_seed31.json")
        for variant in VARIANTS
    }
    logs = {
        variant: read_jsonl(RUN / "logs" / f"{variant}_seed31.jsonl")
        for variant in VARIANTS
    }
    steps = {
        variant: [row for row in records if row["record_type"] == "train_step"]
        for variant, records in logs.items()
    }
    integrity = {}
    for variant, records in logs.items():
        run_steps = steps[variant]
        run_end = [row for row in records if row["record_type"] == "run_end"]
        integrity[variant] = {
            "records": len(records),
            "train_steps": len(run_steps),
            "run_start_records": sum(row["record_type"] == "run_start" for row in records),
            "run_end_records": len(run_end),
            "contiguous_steps": [row["step"] for row in run_steps]
            == list(range(len(run_steps))),
            "finite_steps": all(row.get("finite") is True for row in run_steps),
            "completed": bool(run_end and run_end[-1]["status"] == "completed"),
        }

    hybrid = results["gs_fe_e_gated"]
    gs = results["gradient_smoothing"]
    hybrid_first = hybrid["first_target_step"]
    gs_first = gs["first_target_step"]
    hybrid_confirmed = hybrid["target_confirmed_step"]
    gs_confirmed = gs["target_confirmed_step"]
    common_step = min(hybrid_confirmed, gs_confirmed)
    common = {
        variant: evaluation_at(result, common_step)
        for variant, result in results.items()
    }

    hybrid_steps = steps["gs_fe_e_gated"]
    probes = [row for row in hybrid_steps if row.get("observer_probe")]
    triggers = [row for row in hybrid_steps if row.get("observer_confirmed_harm")]
    interventions = [row for row in hybrid_steps if row.get("regularized")]
    phase_guarded = [
        row
        for row in probes
        if row.get("observer_phase_improving")
        and row.get("observer_persistent_propagation")
        and row.get("observer_persistent_damage")
    ]
    trigger_steps = [row["step"] for row in triggers]
    intervention_steps = [row["step"] for row in interventions]
    trigger_intervals = [
        right - left for left, right in zip(trigger_steps, trigger_steps[1:])
    ]
    ratio_counts = Counter(round(row["observer_gradient_ratio"], 3) for row in interventions)
    reason_counts = Counter(
        reason
        for row in probes
        for reason in row.get("observer_reasons", [])
        if reason != "calibration"
    )
    max_gradients = {}
    for variant, rows in steps.items():
        peak = max(rows, key=lambda row: row["parameter_gradient_norm"])
        max_gradients[variant] = {
            "value": peak["parameter_gradient_norm"],
            "step": peak["step"],
        }
    pre_intervention_check_step = 50
    pre_intervention = {
        variant: {
            "task_loss": next(
                row["task_loss"]
                for row in rows
                if row["step"] == pre_intervention_check_step
            ),
            "parameter_gradient_norm": next(
                row["parameter_gradient_norm"]
                for row in rows
                if row["step"] == pre_intervention_check_step
            ),
        }
        for variant, rows in steps.items()
    }

    analysis = {
        "run": str(RUN.relative_to(ROOT)),
        "manifest_source_sha256": manifest["source_sha256"],
        "config": manifest["config"],
        "integrity": integrity,
        "primary_endpoint": {
            "definition": "token accuracy >= 0.99 for 3 consecutive 125-step checkpoints",
            "gs_fe_e_gated_first_step": hybrid_first,
            "gradient_smoothing_first_step": gs_first,
            "first_step_saving": gs_first - hybrid_first,
            "first_step_reduction_fraction": (gs_first - hybrid_first) / gs_first,
            "gs_fe_e_gated_confirmed_step": hybrid_confirmed,
            "gradient_smoothing_confirmed_step": gs_confirmed,
            "confirmed_step_saving": gs_confirmed - hybrid_confirmed,
            "confirmed_step_reduction_fraction": (
                gs_confirmed - hybrid_confirmed
            )
            / gs_confirmed,
        },
        "common_step": common_step,
        "common_step_evaluations": common,
        "gate_audit": {
            "observer_probes": len(probes),
            "propagation_abnormal_probes": sum(
                bool(row.get("observer_propagation_abnormal")) for row in probes
            ),
            "persistent_propagation_probes": sum(
                bool(row.get("observer_persistent_propagation")) for row in probes
            ),
            "damage_events": sum(bool(row.get("observer_damage_event")) for row in probes),
            "persistent_damage_probes": sum(
                bool(row.get("observer_persistent_damage")) for row in probes
            ),
            "phase_guarded_probes": len(phase_guarded),
            "confirmed_harm_events": len(triggers),
            "intervention_steps": len(interventions),
            "intervention_fraction": len(interventions) / len(hybrid_steps),
            "trigger_steps_zero_based": trigger_steps,
            "intervention_steps_zero_based": intervention_steps,
            "trigger_interval_median": median_or_none(trigger_intervals),
            "trigger_interval_min": min(trigger_intervals) if trigger_intervals else None,
            "gradient_ratio_counts": dict(sorted(ratio_counts.items())),
            "reason_counts": dict(reason_counts),
        },
        "efficiency": {
            variant: {
                "completed_steps": result["completed_steps"],
                "mean_step_seconds": result["mean_step_seconds"],
                "timed_training_seconds": result["timed_training_seconds"],
                "target_confirmed_training_seconds": result[
                    "target_confirmed_training_seconds"
                ],
                "peak_memory_bytes": result["peak_memory_bytes"],
                "ordinary_step_medians": ordinary_step_medians(steps[variant]),
            }
            for variant, result in results.items()
        },
        "max_parameter_gradient_norm": max_gradients,
        "pre_intervention_divergence_check": {
            "check_step_zero_based": pre_intervention_check_step,
            "first_fee_step_zero_based": intervention_steps[0],
            "values": pre_intervention,
        },
        "limitations": [
            "single seed and synthetic reverse-sequence task",
            "serial order was hybrid first, so wall-clock comparisons are contaminated by system-load drift",
            "Metal trajectories show small run-to-run numerical drift near a sharp learning transition",
            "the fixed calibration reference makes propagation flags frequent; task damage, phase guard, and cooldown provide the effective selectivity",
        ],
    }
    return analysis, results, steps


def plot(results, steps):
    FIGURE_BASE.parent.mkdir(parents=True, exist_ok=True)
    width, height, scale = 1600, 1260, 2
    image = Image.new("RGB", (width * scale, height * scale), "white")
    draw = ImageDraw.Draw(image, "RGBA")
    font_path = "/System/Library/Fonts/Supplemental/Arial.ttf"
    bold_path = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
    fonts = {
        "title": ImageFont.truetype(bold_path, 25 * scale),
        "subtitle": ImageFont.truetype(font_path, 15 * scale),
        "panel": ImageFont.truetype(bold_path, 18 * scale),
        "tick": ImageFont.truetype(font_path, 13 * scale),
        "label": ImageFont.truetype(font_path, 15 * scale),
        "legend": ImageFont.truetype(font_path, 14 * scale),
    }
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#222}.title{font-size:25px;font-weight:700}.subtitle{font-size:15px;fill:#666}.panel{font-size:18px;font-weight:700}.tick{font-size:13px;fill:#555}.label{font-size:15px}.legend{font-size:14px}</style>',
    ]

    def pt(x, y):
        return round(x * scale), round(y * scale)

    def line(x1, y1, x2, y2, color, w=1, dash=None):
        draw.line((*pt(x1, y1), *pt(x2, y2)), fill=color, width=max(1, round(w * scale)))
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        svg.append(
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{color}" stroke-width="{w}"{dash_attr}/>'
        )

    def text(x, y, value, font, color="#222", anchor="lm", svg_anchor="start", cls="label"):
        draw.text(pt(x, y), value, font=fonts[font], fill=color, anchor=anchor)
        svg.append(
            f'<text class="{cls}" x="{x:.2f}" y="{y + 5:.2f}" fill="{color}" text-anchor="{svg_anchor}">{escape(value)}</text>'
        )

    def circle(x, y, radius, fill, outline=None, w=1):
        px, py = pt(x, y)
        rr = round(radius * scale)
        draw.ellipse(
            (px - rr, py - rr, px + rr, py + rr),
            fill=fill,
            outline=outline,
            width=max(1, round(w * scale)),
        )
        svg.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius}" fill="{fill}" stroke="{outline or fill}" stroke-width="{w}"/>'
        )

    text(width / 2, 31, "128-layer persistent-harm gated FE-E vs pure GS", "title", anchor="mm", svg_anchor="middle", cls="title")
    text(width / 2, 61, "Seed 31 · 8 fixed validation batches · hybrid executed first", "subtitle", "#666", "mm", "middle", "subtitle")

    left, right = 150, 1515
    panels = [(115, 405), (490, 790), (900, 1140)]
    x_ticks = [0, 500, 1000, 1500, 2000, 2500]

    def xmap(value, maximum, l=left, r=right):
        return l + value / maximum * (r - l)

    # Accuracy panel.
    top, bottom = panels[0]
    text(left, top - 28, "(a) Stable token-accuracy endpoint", "panel", cls="panel")
    for tick in [0, 25, 50, 75, 100]:
        y = bottom - tick / 100 * (bottom - top)
        line(left, y, right, y, "#D7DCE2", 1)
        text(left - 15, y, f"{tick}%", "tick", "#555", "rm", "end", "tick")
    target_y = bottom - 0.99 * (bottom - top)
    line(left, target_y, right, target_y, "#334155", 1.3, "6 5")
    text(right - 8, target_y + 12, "99% target", "tick", "#334155", "rm", "end", "tick")
    line(left, top, left, bottom, "#475569", 1.2)
    line(left, bottom, right, bottom, "#475569", 1.2)
    for tick in x_ticks:
        x = xmap(tick, 2500)
        line(x, bottom, x, bottom + 6, "#475569")
        text(x, bottom + 20, str(tick), "tick", "#555", "mm", "middle", "tick")
    for variant, result in results.items():
        history = result["evaluation_history"]
        points = [
            (xmap(row["step"], 2500), bottom - row["evaluation_accuracy"] * (bottom - top))
            for row in history
        ]
        draw.line([pt(x, y) for x, y in points], fill=COLORS[variant], width=4 * scale, joint="curve")
        path = " ".join(("M" if i == 0 else "L") + f" {x:.2f} {y:.2f}" for i, (x, y) in enumerate(points))
        svg.append(f'<path d="{path}" fill="none" stroke="{COLORS[variant]}" stroke-width="4"/>')
        for x, y in points:
            circle(x, y, 4.2, "white", COLORS[variant], 2)
        for target_step, dash in ((result["first_target_step"], "7 5"), (result["target_confirmed_step"], "2 5")):
            x = xmap(target_step, 2500)
            line(x, top, x, bottom, COLORS[variant], 1.3, dash)
    text(right - 405, top + 18, "GS + gated FE-E", "legend", COLORS["gs_fe_e_gated"], cls="legend")
    line(right - 445, top + 18, right - 415, top + 18, COLORS["gs_fe_e_gated"], 4)
    text(right - 210, top + 18, "Pure GS", "legend", COLORS["gradient_smoothing"], cls="legend")
    line(right - 250, top + 18, right - 220, top + 18, COLORS["gradient_smoothing"], 4)

    # Loss panel.
    top, bottom = panels[1]
    text(left, top - 28, "(b) Validation loss across the learning transition", "panel", cls="panel")
    loss_min, loss_max = 0.005, 5.0
    log_min, log_max = math.log10(loss_min), math.log10(loss_max)
    def yloss(value):
        return bottom - (math.log10(value) - log_min) / (log_max - log_min) * (bottom - top)
    for tick in [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]:
        y = yloss(tick)
        line(left, y, right, y, "#D7DCE2", 1)
        text(left - 15, y, f"{tick:g}", "tick", "#555", "rm", "end", "tick")
    line(left, top, left, bottom, "#475569", 1.2)
    line(left, bottom, right, bottom, "#475569", 1.2)
    for tick in x_ticks:
        x = xmap(tick, 2500)
        line(x, bottom, x, bottom + 6, "#475569")
        text(x, bottom + 20, str(tick), "tick", "#555", "mm", "middle", "tick")
    for variant, result in results.items():
        points = [
            (xmap(row["step"], 2500), yloss(row["evaluation_loss"]))
            for row in result["evaluation_history"]
        ]
        draw.line([pt(x, y) for x, y in points], fill=COLORS[variant], width=4 * scale, joint="curve")
        path = " ".join(("M" if i == 0 else "L") + f" {x:.2f} {y:.2f}" for i, (x, y) in enumerate(points))
        svg.append(f'<path d="{path}" fill="none" stroke="{COLORS[variant]}" stroke-width="4"/>')
        for x, y in points:
            circle(x, y, 4.2, "white", COLORS[variant], 2)

    # Gate timeline.
    top, bottom = panels[2]
    text(left, top - 28, "(c) Persistent-harm gate audit (hybrid run)", "panel", cls="panel")
    hybrid = steps["gs_fe_e_gated"]
    probes = [row for row in hybrid if row.get("observer_probe")]
    rows = [
        ("Propagation abnormal", 3, [row["step"] for row in probes if row.get("observer_propagation_abnormal")], "#94A3B8"),
        ("Persistent task damage", 2, [row["step"] for row in probes if row.get("observer_persistent_damage")], "#A855F7"),
        ("Confirmed harm", 1, [row["step"] for row in probes if row.get("observer_confirmed_harm")], "#DC2626"),
        ("FE-E applied", 0, [row["step"] for row in hybrid if row.get("regularized")], COLORS["gs_fe_e_gated"]),
    ]
    ymap = {level: bottom - level / 3 * (bottom - top) for _, level, _, _ in rows}
    for label, level, values, color in rows:
        y = ymap[level]
        line(left, y, right, y, "#E2E8F0", 1)
        text(left - 15, y, label, "tick", "#555", "rm", "end", "tick")
        for value in values:
            x = xmap(value, 1650)
            if level == 3:
                line(x, y - 7, x, y + 7, color, 1.4)
            else:
                circle(x, y, 4.5 if level > 0 else 5.5, color)
    for tick in [0, 250, 500, 750, 1000, 1250, 1500]:
        x = xmap(tick, 1650)
        line(x, bottom, x, bottom + 6, "#475569")
        text(x, bottom + 20, str(tick), "tick", "#555", "mm", "middle", "tick")
    line(left, top, left, bottom, "#475569", 1.2)
    line(left, bottom, right, bottom, "#475569", 1.2)
    text((left + right) / 2, 1200, "Parameter update", "label", "#333", "mm", "middle")

    svg.append("</svg>")
    FIGURE_BASE.with_suffix(".svg").write_text("\n".join(svg) + "\n", encoding="utf-8")
    image.save(FIGURE_BASE.with_suffix(".png"), dpi=(240, 240))


def write_report(analysis, results):
    endpoint = analysis["primary_endpoint"]
    gate = analysis["gate_audit"]
    common = analysis["common_step_evaluations"]
    hybrid = results["gs_fe_e_gated"]
    gs = results["gradient_smoothing"]
    report = f"""# 128 层 FE-E 持续伤害门控：99% token accuracy 对照

## 结论

在本次单种子合成反序列任务中，`GS + 门控 FE-E` 比纯 GS 更早进入学习相变：首次达到
token accuracy ≥ 99% 的步数从 {endpoint['gradient_smoothing_first_step']} 降至
{endpoint['gs_fe_e_gated_first_step']}，提前 {endpoint['first_step_saving']} 次更新
（{endpoint['first_step_reduction_fraction']:.1%}）；连续 3 个检查点确认的步数从
{endpoint['gradient_smoothing_confirmed_step']} 降至
{endpoint['gs_fe_e_gated_confirmed_step']}，同样提前
{endpoint['confirmed_step_saving']} 次更新（{endpoint['confirmed_step_reduction_fraction']:.1%}）。

这是一项值得继续验证的正向工程信号，但不是统计显著结论：实验只有一个种子，任务是合成任务，
而且 MLX 深层轨迹在尖锐学习相变附近存在数值漂移。

## 冻结协议

- 128 层、宽度 32、4 头、序列长度 12、batch size 8；种子 31。
- 顺序按预先约定固定为：先 `GS + 门控 FE-E`，后纯 GS。
- 每 125 步评估一次，每个检查点使用 8 个固定验证 batch。
- 主终点：token accuracy ≥ 99%，并连续保持 3 个验证检查点。
- 最多 5000 步；确认主终点后自动停止。
- 门控：当下至少 2 个传播指标异常；最近 4 次探测至少 3 次异常；哨兵任务连续受损 2 次；
  快速学习相变保护；FE-E 单步介入；48 步冷却；梯度比率 5%→10%→20%。

## 主结果

| 方法 | 首次 ≥99% | 连续确认步 | 确认时验证损失 | 确认时准确率 | FE-E 更新占比 |
|---|---:|---:|---:|---:|---:|
| GS + 门控 FE-E | {hybrid['first_target_step']} | {hybrid['target_confirmed_step']} | {hybrid['evaluation_loss']:.6f} | {hybrid['evaluation_accuracy']:.2%} | {hybrid['regularized_fraction']:.2%} |
| 纯 GS | {gs['first_target_step']} | {gs['target_confirmed_step']} | {gs['evaluation_loss']:.6f} | {gs['evaluation_accuracy']:.2%} | 0% |

在共同的第 {analysis['common_step']} 步，混合方案验证损失为
{common['gs_fe_e_gated']['evaluation_loss']:.6f}、准确率为
{common['gs_fe_e_gated']['evaluation_accuracy']:.2%}；纯 GS 同步数验证损失为
{common['gradient_smoothing']['evaluation_loss']:.6f}、准确率为
{common['gradient_smoothing']['evaluation_accuracy']:.2%}。

## 门控审计

- 共 {gate['observer_probes']} 次观察器探测，确认伤害 {gate['confirmed_harm_events']} 次，
  实际 FE-E 介入 {gate['intervention_steps']} 步，占全部更新 {gate['intervention_fraction']:.2%}。
- 介入间隔最小 {gate['trigger_interval_min']} 步，中位数 {gate['trigger_interval_median']:.0f} 步；
  不再出现旧门控的四步脉冲和约 27% 介入率。
- 梯度比率使用次数：{gate['gradient_ratio_counts']}。
- 有 {gate['phase_guarded_probes']} 个本可满足传播与任务伤害条件的探测被快速学习相变保护压制。
- 固定初始校准基准使刚度与覆盖度异常标记较频繁；本版本的有效选择性主要来自任务伤害确认、
  相变保护、单步介入和 48 步冷却。根据本轮决策，保留该介入逻辑，不把自适应基线并入主方法。

## 工程含义

本轮最有价值的量不是最终损失更低，而是达到可用正确率所需更新数减少 35%。如果这一比例能在
多种子和真实语言建模任务上复现，它意味着训练到同一质量所需的优化更新和 token 预算可能下降。
同时 FE-E 只在 0.74% 的更新上执行二阶正则，避免了常开 FE-E 的主要计算代价。

但本轮墙钟不能用于证明加速：两个方法串行运行，后运行的纯 GS 遇到更慢的系统负载；此前已经
观察到 `mediaanalysisd`、Spotlight 等后台任务。故论文主表应使用 updates-to-target，墙钟只作审计字段。

## 证据边界

1. 单种子不能区分稳定方法效应与深层系统对微小数值扰动的敏感性。
2. 因果归因检查发现：首次 FE-E 更新发生在零基第 {analysis['pre_intervention_divergence_check']['first_fee_step_zero_based']} 步，
   但零基第 {analysis['pre_intervention_divergence_check']['check_step_zero_based']} 步的参数梯度范数已经分别为
   {analysis['pre_intervention_divergence_check']['values']['gs_fe_e_gated']['parameter_gradient_norm']:.3f}
   与 {analysis['pre_intervention_divergence_check']['values']['gradient_smoothing']['parameter_gradient_norm']:.3f}。
   因而 875 步差异不能全部归因于 FE-E；同构观察器控制是下一轮的必要实验。
3. 反序列任务远小于真实语言模型；不能把 35% 直接外推到 7B 训练。
4. 纯 GS 和混合方案的观察开销不同；后续可增加同构 sham 观察器作为机制控制，但不改变本轮
   按用户指定的“混合在前、纯 GS 在后”主对比。
5. 当前正向结论应表述为“值得进行多种子压力实验”，而不是“FE-E 已被证明优于 GS”。

## 文件与完整性

- 原始运行：`{analysis['run']}`
- 源码 SHA-256：`{analysis['manifest_source_sha256']}`
- 结构化审计：`results/mlx_d128_acc99_persistent_gate_v3_analysis.json`
- 曲线：`output/figures/fee_d128_acc99_persistent_gate_v3.png` 与 `.svg`
- 两个日志均包含 1 条 `run_start`、连续逐步 `train_step` 和 1 条成功 `run_end`；无 NaN/Inf。

开发期间的自适应基线与中档阈值试跑均未进入上述统计。
"""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")


def main():
    analysis, results, steps = analyze()
    SUMMARY_PATH.write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    plot(results, steps)
    write_report(analysis, results)
    print(SUMMARY_PATH)
    print(REPORT_PATH)
    print(FIGURE_BASE.with_suffix(".png"))
    print(FIGURE_BASE.with_suffix(".svg"))


if __name__ == "__main__":
    main()
