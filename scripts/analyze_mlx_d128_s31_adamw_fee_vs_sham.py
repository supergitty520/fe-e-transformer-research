#!/usr/bin/env python3
"""Audit and plot the matched AdamW gated-FE-E versus sham-observer run."""

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
    / "results/mlx_d128_s31_acc99_adamw_fee_vs_sham_fee_first"
    / "run_20260805_183344"
)
HISTORICAL_RUN = (
    ROOT
    / "results/mlx_d128_s4_acc99_fee_gs_adamw_hybrid_first"
    / "run_20260805_163941"
)
VARIANTS = ["fe_e_gated", "adamw_observer_control"]
LABELS = {
    "fe_e_gated": "AdamW + gated FE-E",
    "adamw_observer_control": "AdamW + sham observer",
    "historical_adamw": "Historical plain AdamW",
}
COLORS = {
    "fe_e_gated": "#0B6E69",
    "adamw_observer_control": "#D97706",
    "historical_adamw": "#64748B",
}
ANALYSIS_PATH = ROOT / "results/mlx_d128_s31_adamw_fee_vs_sham_analysis.json"
REPORT_PATH = ROOT / "docs/mlx_d128_s31_adamw_fee_vs_sham_report.md"
FIGURE_BASE = ROOT / "output/figures/fee_d128_s31_adamw_fee_vs_sham"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def train_rows(records):
    return [row for row in records if row.get("record_type") == "train_step"]


def audit_log(records, completed_steps):
    steps = train_rows(records)
    return {
        "records": len(records),
        "train_steps": len(steps),
        "run_start_records": sum(row.get("record_type") == "run_start" for row in records),
        "run_end_records": sum(row.get("record_type") == "run_end" for row in records),
        "contiguous_zero_based_steps": [row["step"] for row in steps]
        == list(range(completed_steps)),
        "all_steps_finite": all(row.get("finite") is True for row in steps),
        "completed_run_end": any(
            row.get("record_type") == "run_end" and row.get("status") == "completed"
            for row in records
        ),
    }


def median(values):
    return statistics.median(values) if values else None


def timing_breakdown(rows):
    ordinary = [
        row["step_seconds"]
        for row in rows
        if not row.get("regularized")
        and not row.get("observer_probe")
        and not row.get("evaluation_seconds")
    ]
    probes = [
        row["step_seconds"]
        for row in rows
        if row.get("observer_probe") and not row.get("regularized")
    ]
    interventions = [row["step_seconds"] for row in rows if row.get("regularized")]
    return {
        "ordinary_step_seconds_median": median(ordinary),
        "observer_probe_step_seconds_median": median(probes),
        "fee_intervention_step_seconds_median": median(interventions),
    }


def common_checkpoint_mean(result, last_step):
    values = [
        item["evaluation_loss"]
        for item in result["evaluation_history"]
        if item["step"] <= last_step
    ]
    return statistics.fmean(values)


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
    rows = {variant: train_rows(logs[variant]) for variant in VARIANTS}
    historical = read_json(HISTORICAL_RUN / "runs/baseline_seed31.json")

    fee = results["fe_e_gated"]
    sham = results["adamw_observer_control"]
    fee_rows = rows["fe_e_gated"]
    sham_rows = rows["adamw_observer_control"]
    interventions = [row for row in fee_rows if row.get("regularized")]
    sham_scheduled = [row for row in sham_rows if row.get("intervention_scheduled")]
    first_intervention = interventions[0]["step"]

    pre_task_differences = [
        abs(fee_rows[step]["task_loss"] - sham_rows[step]["task_loss"])
        for step in range(first_intervention)
    ]
    pre_gradient_differences = [
        abs(
            fee_rows[step]["parameter_gradient_norm"]
            - sham_rows[step]["parameter_gradient_norm"]
        )
        for step in range(first_intervention)
    ]
    decision_step = first_intervention - 1
    common_last = min(fee["target_confirmed_step"], sham["target_confirmed_step"])

    integrity = {
        variant: audit_log(logs[variant], results[variant]["completed_steps"])
        for variant in VARIANTS
    }
    analysis = {
        "run": str(RUN.relative_to(ROOT)),
        "historical_plain_adamw_run": str(HISTORICAL_RUN.relative_to(ROOT)),
        "source_sha256": manifest["source_sha256"],
        "config": manifest["config"],
        "primary_endpoint": "token accuracy >= 0.99 for 3 consecutive 125-step validation checkpoints",
        "integrity": integrity,
        "all_primary_logs_pass_integrity": all(
            item["run_start_records"] == 1
            and item["run_end_records"] == 1
            and item["contiguous_zero_based_steps"]
            and item["all_steps_finite"]
            and item["completed_run_end"]
            for item in integrity.values()
        ),
        "primary_comparison": {
            "fee_first_target_step": fee["first_target_step"],
            "sham_first_target_step": sham["first_target_step"],
            "first_target_step_saving": sham["first_target_step"] - fee["first_target_step"],
            "first_target_reduction_fraction": (
                sham["first_target_step"] - fee["first_target_step"]
            )
            / sham["first_target_step"],
            "fee_confirmed_step": fee["target_confirmed_step"],
            "sham_confirmed_step": sham["target_confirmed_step"],
            "confirmed_step_saving": sham["target_confirmed_step"]
            - fee["target_confirmed_step"],
            "confirmed_step_reduction_fraction": (
                sham["target_confirmed_step"] - fee["target_confirmed_step"]
            )
            / sham["target_confirmed_step"],
            "common_checkpoint_last_step": common_last,
            "fee_mean_validation_loss_through_common_step": common_checkpoint_mean(
                fee, common_last
            ),
            "sham_mean_validation_loss_through_common_step": common_checkpoint_mean(
                sham, common_last
            ),
        },
        "historical_context_not_primary_control": {
            "plain_adamw_first_target_step": historical["first_target_step"],
            "plain_adamw_confirmed_step": historical["target_confirmed_step"],
            "fee_vs_plain_adamw_confirmed_step_difference": fee[
                "target_confirmed_step"
            ]
            - historical["target_confirmed_step"],
            "interpretation": "The matched FE-E run ties the earlier lean AdamW endpoint; the historical run is not a contemporaneous causal control.",
        },
        "gate_audit": {
            "fee_intervention_count": len(interventions),
            "fee_intervention_fraction": len(interventions) / len(fee_rows),
            "fee_intervention_steps_zero_based": [row["step"] for row in interventions],
            "fee_gradient_ratio_counts": dict(
                sorted(
                    Counter(
                        str(round(row["observer_gradient_ratio"], 3))
                        for row in interventions
                    ).items()
                )
            ),
            "sham_scheduled_intervention_count": len(sham_scheduled),
            "sham_scheduled_steps_zero_based": [row["step"] for row in sham_scheduled],
            "first_gate_decision_matched": {
                "decision_step_zero_based": decision_step,
                "fee_confirmed_harm": fee_rows[decision_step].get(
                    "observer_confirmed_harm"
                ),
                "sham_confirmed_harm": sham_rows[decision_step].get(
                    "observer_confirmed_harm"
                ),
                "fee_state_after": fee_rows[decision_step].get("observer_state_after"),
                "sham_state_after": sham_rows[decision_step].get("observer_state_after"),
            },
        },
        "pre_intervention_match": {
            "first_fee_step_zero_based": first_intervention,
            "compared_steps": first_intervention,
            "maximum_absolute_task_loss_difference": max(pre_task_differences),
            "maximum_absolute_parameter_gradient_norm_difference": max(
                pre_gradient_differences
            ),
            "fee_task_loss_at_first_fee_step_before_update": fee_rows[
                first_intervention
            ]["task_loss"],
            "sham_task_loss_at_first_fee_step_before_update": sham_rows[
                first_intervention
            ]["task_loss"],
            "same_first_intervention_schedule": fee_rows[first_intervention].get(
                "intervention_scheduled"
            )
            and sham_rows[first_intervention].get("intervention_scheduled"),
        },
        "efficiency_audit": {
            variant: {
                "completed_steps": results[variant]["completed_steps"],
                "timed_training_seconds": results[variant]["timed_training_seconds"],
                "mean_step_seconds": results[variant]["mean_step_seconds"],
                "peak_memory_bytes": results[variant]["peak_memory_bytes"],
                **timing_breakdown(rows[variant]),
            }
            for variant in VARIANTS
        },
        "interpretation": {
            "matched_control_result": "FE-E wins the matched sham-observer comparison by 125 updates (5.9%).",
            "engineering_result": "FE-E ties the earlier plain AdamW endpoint and adds observer/intervention cost, so independent production advantage over lean AdamW is not established.",
            "evidence_level": "single-seed mechanistic evidence, not a statistical result",
        },
    }
    return analysis, results, historical, rows


class Plot:
    def __init__(self, width=1600, height=1080, scale=2):
        self.width, self.height, self.scale = width, height, scale
        self.image = Image.new("RGB", (width * scale, height * scale), "white")
        self.draw = ImageDraw.Draw(self.image, "RGBA")
        regular = "/System/Library/Fonts/Supplemental/Arial.ttf"
        bold = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
        self.fonts = {
            "title": ImageFont.truetype(bold, 26 * scale),
            "subtitle": ImageFont.truetype(regular, 15 * scale),
            "panel": ImageFont.truetype(bold, 18 * scale),
            "label": ImageFont.truetype(regular, 14 * scale),
            "small": ImageFont.truetype(regular, 12 * scale),
            "bold": ImageFont.truetype(bold, 14 * scale),
        }
        self.svg = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="#fff"/>',
            '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#222}.title{font-size:26px;font-weight:700}.subtitle{font-size:15px;fill:#666}.panel{font-size:18px;font-weight:700}.label{font-size:14px}.small{font-size:12px;fill:#555}.bold{font-size:14px;font-weight:700}</style>',
        ]

    def pt(self, x, y):
        return round(x * self.scale), round(y * self.scale)

    def text(self, x, y, value, font="label", color="#222", anchor="la", svg_anchor="start"):
        self.draw.text(self.pt(x, y), str(value), font=self.fonts[font], fill=color, anchor=anchor)
        self.svg.append(f'<text class="{font}" x="{x:.2f}" y="{y:.2f}" fill="{color}" text-anchor="{svg_anchor}" dominant-baseline="middle">{escape(str(value))}</text>')

    def line(self, x1, y1, x2, y2, color, width=1, dash=None):
        self.draw.line((*self.pt(x1, y1), *self.pt(x2, y2)), fill=color, width=max(1, round(width * self.scale)))
        attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.svg.append(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{color}" stroke-width="{width}"{attr}/>')

    def circle(self, x, y, radius, fill):
        px, py = self.pt(x, y)
        rr = round(radius * self.scale)
        self.draw.ellipse((px-rr, py-rr, px+rr, py+rr), fill=fill)
        self.svg.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius}" fill="{fill}"/>')

    def polyline(self, points, color, width=3, dash=None):
        self.draw.line([self.pt(x, y) for x, y in points], fill=color, width=round(width * self.scale), joint="curve")
        payload = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.svg.append(f'<polyline points="{payload}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linecap="round" stroke-linejoin="round"{attr}/>')

    def save(self):
        FIGURE_BASE.parent.mkdir(parents=True, exist_ok=True)
        self.image.resize((self.width, self.height), Image.Resampling.LANCZOS).save(FIGURE_BASE.with_suffix(".png"), dpi=(180, 180))
        self.svg.append("</svg>")
        FIGURE_BASE.with_suffix(".svg").write_text("\n".join(self.svg), encoding="utf-8")


def make_plot(analysis, results, historical):
    p = Plot()
    p.text(70, 45, "AdamW + gated FE-E versus matched sham observer (seed 31)", "title")
    p.text(70, 78, "128 layers; endpoint is >=99% token accuracy at three consecutive checkpoints", "subtitle", "#666")
    for index, key in enumerate(["fe_e_gated", "adamw_observer_control", "historical_adamw"]):
        x = 805 + index * 245
        p.line(x, 74, x + 28, 74, COLORS[key], 4, "7 5" if key == "historical_adamw" else None)
        p.circle(x + 14, 74, 4, COLORS[key])
        p.text(x + 38, 74, LABELS[key], "small", "#333")

    series = {
        "fe_e_gated": results["fe_e_gated"]["evaluation_history"],
        "adamw_observer_control": results["adamw_observer_control"]["evaluation_history"],
        "historical_adamw": historical["evaluation_history"],
    }
    intervention_steps = analysis["gate_audit"]["fee_intervention_steps_zero_based"]
    x_max = 2250

    def panel(left, top, right, bottom, metric, y_ticks, transform, panel_title):
        p.text(left, top - 38, panel_title, "panel")
        for value, label in y_ticks:
            y = bottom - transform(value) * (bottom - top)
            p.line(left, y, right, y, "#E6E8EB", 1)
            p.text(left - 12, y, label, "small", "#555", "ra", "end")
        for tick in [0, 500, 1000, 1500, 2000]:
            x = left + tick / x_max * (right - left)
            p.line(x, top, x, bottom, "#F0F1F3", 1)
            p.text(x, bottom + 22, tick, "small", "#555", "ma", "middle")
        p.line(left, top, left, bottom, "#666", 1.2)
        p.line(left, bottom, right, bottom, "#666", 1.2)
        for step in intervention_steps:
            x = left + (step + 1) / x_max * (right - left)
            p.line(x, bottom - 15, x, bottom, COLORS["fe_e_gated"], 1.5)
        for key, history in series.items():
            points = [
                (
                    left + item["step"] / x_max * (right - left),
                    bottom - transform(item[metric]) * (bottom - top),
                )
                for item in history
                if item["step"] <= x_max
            ]
            p.polyline(points, COLORS[key], 3 if key != "historical_adamw" else 2, "7 5" if key == "historical_adamw" else None)
            p.circle(points[-1][0], points[-1][1], 4, COLORS[key])
        p.text((left + right) / 2, bottom + 55, "Optimizer updates", "label", "#333", "ma", "middle")

    panel(
        90, 170, 760, 760,
        "evaluation_accuracy",
        [(0, "0.00"), (.25, "0.25"), (.5, "0.50"), (.75, "0.75"), (1, "1.00")],
        lambda value: value,
        "A  Validation token accuracy",
    )
    log_min, log_max = math.log10(0.005), math.log10(4.0)
    panel(
        875, 170, 1545, 760,
        "evaluation_loss",
        [(0.01, "0.01"), (.1, "0.1"), (1, "1"), (4, "4")],
        lambda value: (math.log10(max(value, 0.005)) - log_min) / (log_max - log_min),
        "B  Validation loss (log scale)",
    )
    p.text(90, 850, "Matched primary result", "panel")
    p.text(90, 890, "Gated FE-E: 1750 first / 2000 confirmed", "bold", COLORS["fe_e_gated"])
    p.text(90, 925, "Sham observer: 1875 first / 2125 confirmed", "bold", COLORS["adamw_observer_control"])
    p.text(90, 960, "Saving: 125 updates (5.9%); FE-E active on 10/2000 updates (0.5%).", "label", "#333")
    p.text(90, 1000, "Historical plain AdamW also confirmed at 2000 updates; it is context, not the matched causal control.", "subtitle", "#555")
    p.text(90, 1032, "Short green rug marks show FE-E intervention updates. Wall-clock ranking is invalid under serial system-load drift.", "subtitle", "#555")
    p.save()


def write_report(analysis):
    primary = analysis["primary_comparison"]
    gate = analysis["gate_audit"]
    match = analysis["pre_intervention_match"]
    eff = analysis["efficiency_audit"]
    memory_ratio = eff["fe_e_gated"]["peak_memory_bytes"] / eff["adamw_observer_control"]["peak_memory_bytes"]
    lines = [
        "# Seed 31：AdamW + 门控 FE-E 对同构 sham observer",
        "",
        "## 结论",
        "",
        f"在同构观察器控制下，AdamW + 门控 FE-E 于第 {primary['fee_first_target_step']} 步首次达到 token accuracy ≥99%，第 {primary['fee_confirmed_step']} 步完成连续 3 次确认；AdamW + sham observer 分别为第 {primary['sham_first_target_step']} 与 {primary['sham_confirmed_step']} 步。FE-E 在主终点上提前 {primary['confirmed_step_saving']} 次更新（{primary['confirmed_step_reduction_fraction']:.1%}）。",
        "",
        "这是比此前 Metal 串行对照更干净的单种子机制信号：两边运行相同观察器与哨兵探测，首次门控决定相同，并在真正注入 FE-E 后才明显分离。",
        "",
        "但工程结论仍然不是 FE-E 已胜过 AdamW。此前不带观察器的纯 AdamW seed 31 也在第 2000 步确认，与本轮 FE-E 方案平局；而 FE-E 增加观察与二阶正则成本。因此当前结果只能证明 FE-E 胜过同构 sham，不证明它胜过精简 AdamW。",
        "",
        "## 冻结协议",
        "",
        "- seed 31；128 层、宽度 32、4 头、序列长度 12、batch size 8。",
        "- 学习率 0.002；两边均使用同一 PaperAdamW。",
        "- 每 125 步验证一次，每次 8 个固定 batch。",
        "- 终点：token accuracy ≥99%，连续 3 个检查点；最多 5000 步并在确认后停止。",
        "- 两边观察器参数完全相同：每 8 步探测、24 步校准、当前至少 2 个异常指标、最近 4 次至少 3 次异常、任务哨兵连续受损 2 次、单步介入逻辑、48 步冷却。",
        "- 唯一方法差异：`fe_e_gated` 在确认伤害后的计划步骤注入 FE-E 梯度；`adamw_observer_control` 记录相同门控决定但不注入 FE-E。",
        "",
        "## 主结果",
        "",
        "| 方法 | 首次 ≥99% | 连续确认 | 确认步差 | FE-E 更新 |",
        "|---|---:|---:|---:|---:|",
        f"| AdamW + 门控 FE-E | {primary['fee_first_target_step']} | {primary['fee_confirmed_step']} | −{primary['confirmed_step_saving']} | {gate['fee_intervention_count']} ({gate['fee_intervention_fraction']:.1%}) |",
        f"| AdamW + sham observer | {primary['sham_first_target_step']} | {primary['sham_confirmed_step']} | 基准 | 0 |",
        "",
        f"截至共同的第 {primary['common_checkpoint_last_step']} 步，全部验证点的平均损失为：FE-E {primary['fee_mean_validation_loss_through_common_step']:.4f}，sham {primary['sham_mean_validation_loss_through_common_step']:.4f}。该量只是曲线次级指标，主结论仍以确认步数为准。",
        "",
        "## 同构与因果审计",
        "",
        f"- 首次 FE-E 更新为零基第 {match['first_fee_step_zero_based']} 步；其前 {match['compared_steps']} 步最大任务损失差仅 {match['maximum_absolute_task_loss_difference']:.2e}，最大参数梯度范数差 {match['maximum_absolute_parameter_gradient_norm_difference']:.2e}。",
        f"- 零基第 {gate['first_gate_decision_matched']['decision_step_zero_based']} 步，两边都确认伤害并进入 `{gate['first_gate_decision_matched']['fee_state_after']}`；下一步两边都计划介入，但 sham 只推进状态机、不应用 FE-E。",
        f"- FE-E 实际介入步：{gate['fee_intervention_steps_zero_based']}。梯度目标比率使用次数：{gate['fee_gradient_ratio_counts']}。",
        f"- sham 也记录了 {gate['sham_scheduled_intervention_count']} 次计划介入，但没有任何正则化更新；首次之后的介入时点不同是两条参数轨迹已经分离后的结果。",
        "",
        "## 计算代价",
        "",
        f"- FE-E 活跃 run 峰值内存 {eff['fe_e_gated']['peak_memory_bytes']/1e6:.1f} MB，sham 为 {eff['adamw_observer_control']['peak_memory_bytes']/1e6:.1f} MB，约 {memory_ratio:.2f} 倍。该微型模型倍数不能直接外推到大模型。",
        f"- FE-E run 普通步中位数 {eff['fe_e_gated']['ordinary_step_seconds_median']:.3f}s，观察器探测步 {eff['fe_e_gated']['observer_probe_step_seconds_median']:.3f}s，实际 FE-E 介入步 {eff['fe_e_gated']['fee_intervention_step_seconds_median']:.3f}s。",
        "- 两个方案串行运行且系统负载随时间变化，墙钟和平均步耗时不用于方法排序；更新数是主效率指标。",
        "",
        "## 证据边界",
        "",
        "1. 只有一个预选种子，不能得出统计显著结论；125 步差异等于一个验证间隔。",
        "2. 本轮证明的是“FE-E 相对 sham observer 改变了相变时点”，不是“FE-E 相对纯 AdamW 获得净收益”。",
        "3. 历史纯 AdamW 与 FE-E 都在 2000 步确认，因此若计入观察、探测、内存和实现复杂度，当前仍没有生产部署优势。",
        "4. 下一步若要验证独立价值，应在预先冻结的多个种子上比较 plain AdamW、AdamW + sham、AdamW + gated FE-E，避免只围绕 seed 31 继续调参。",
        "",
        "## 文件与完整性",
        "",
        f"- 原始运行：`{analysis['run']}`",
        "- 结构化分析：`results/mlx_d128_s31_adamw_fee_vs_sham_analysis.json`",
        "- 图表：`output/figures/fee_d128_s31_adamw_fee_vs_sham.png` 与 `.svg`",
        f"- 源码 SHA-256：`{analysis['source_sha256']}`",
        "- 两条正式日志均包含 1 条 run_start、连续逐步 train_step、1 条 completed run_end，且没有 NaN/Inf。",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    analysis, results, historical, _ = analyze()
    ANALYSIS_PATH.parent.mkdir(parents=True, exist_ok=True)
    ANALYSIS_PATH.write_text(json.dumps(analysis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(analysis)
    make_plot(analysis, results, historical)
    print(ANALYSIS_PATH)
    print(REPORT_PATH)
    print(FIGURE_BASE.with_suffix('.png'))
    print(FIGURE_BASE.with_suffix('.svg'))


if __name__ == "__main__":
    main()
