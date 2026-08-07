#!/usr/bin/env python3
"""Audit and plot the 192-layer GS-sham versus GSF LR-shock experiment."""

from __future__ import annotations

import json
import math
from pathlib import Path
import statistics
from xml.sax.saxutils import escape

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
RUN = (
    ROOT
    / "results/mlx_d192_s47_acc99_gs_sham_vs_gsf_lrshock"
    / "run_20260805_194235"
)
VARIANTS = ["gs_observer_control", "gs_fe_e_gated"]
LABELS = {
    "gs_observer_control": "GS-SHAM",
    "gs_fe_e_gated": "GSF",
}
COLORS = {
    "gs_observer_control": "#D97706",
    "gs_fe_e_gated": "#0B6E69",
}
ANALYSIS_PATH = (
    ROOT / "results/mlx_d192_s47_acc99_gs_sham_vs_gsf_lrshock_analysis.json"
)
REPORT_PATH = ROOT / "docs/mlx_d192_s47_acc99_gs_sham_vs_gsf_lrshock_report.md"
FIGURE_BASE = ROOT / "output/figures/fee_d192_s47_gs_sham_vs_gsf_lrshock"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def train_rows(records):
    return [row for row in records if row.get("record_type") == "train_step"]


def audit_log(records, completed_steps):
    steps = train_rows(records)
    return {
        "records": len(records),
        "train_steps": len(steps),
        "run_start_records": sum(
            row.get("record_type") == "run_start" for row in records
        ),
        "run_end_records": sum(
            row.get("record_type") == "run_end" for row in records
        ),
        "contiguous_zero_based_steps": [row["step"] for row in steps]
        == list(range(completed_steps)),
        "all_steps_finite": all(row.get("finite") is True for row in steps),
        "completed_run_end": any(
            row.get("record_type") == "run_end"
            and row.get("status") == "completed"
            for row in records
        ),
    }


def median(values):
    return statistics.median(values) if values else None


def timing_breakdown(rows):
    warm = [row for row in rows if row["step"] >= 5]
    ordinary = [
        row["step_seconds"]
        for row in warm
        if not row.get("adjoint_probe") and not row.get("regularized")
    ]
    probes = [
        row["step_seconds"]
        for row in warm
        if row.get("adjoint_probe") and not row.get("regularized")
    ]
    interventions = [
        row["step_seconds"] for row in warm if row.get("regularized")
    ]
    return {
        "ordinary_count": len(ordinary),
        "probe_count": len(probes),
        "intervention_count": len(interventions),
        "ordinary_step_seconds_median": median(ordinary),
        "observer_probe_step_seconds_median": median(probes),
        "fee_intervention_step_seconds_median": median(interventions),
        "sum_step_seconds": sum(row["step_seconds"] for row in rows),
        "sum_evaluation_seconds": sum(
            row.get("evaluation_seconds", 0.0) for row in rows
        ),
    }


def analyze():
    manifest = read_json(RUN / "manifest.json")
    results = {
        variant: read_json(RUN / "runs" / f"{variant}_seed47.json")
        for variant in VARIANTS
    }
    logs = {
        variant: read_jsonl(RUN / "logs" / f"{variant}_seed47.jsonl")
        for variant in VARIANTS
    }
    rows = {variant: train_rows(logs[variant]) for variant in VARIANTS}
    sham = results["gs_observer_control"]
    gsf = results["gs_fe_e_gated"]
    sham_rows = rows["gs_observer_control"]
    gsf_rows = rows["gs_fe_e_gated"]
    interventions = [row for row in gsf_rows if row.get("regularized")]
    sham_scheduled = [
        row for row in sham_rows if row.get("intervention_scheduled")
    ]
    first_fee_step = interventions[0]["step"]
    pre_steps = range(first_fee_step)
    pre_loss_difference = max(
        abs(gsf_rows[step]["task_loss"] - sham_rows[step]["task_loss"])
        for step in pre_steps
    )
    pre_gradient_difference = max(
        abs(
            gsf_rows[step]["parameter_gradient_norm"]
            - sham_rows[step]["parameter_gradient_norm"]
        )
        for step in pre_steps
    )
    timing = {
        variant: timing_breakdown(rows[variant]) for variant in VARIANTS
    }
    ordinary_speed_ratio = (
        timing["gs_fe_e_gated"]["ordinary_step_seconds_median"]
        / timing["gs_observer_control"]["ordinary_step_seconds_median"]
    )
    observed_time_ratio = (
        gsf["timed_training_seconds"] / sham["timed_training_seconds"]
    )
    normalized_time_ratio = observed_time_ratio / ordinary_speed_ratio
    fee_increment_per_event = (
        timing["gs_fe_e_gated"]["fee_intervention_step_seconds_median"]
        - timing["gs_fe_e_gated"]["observer_probe_step_seconds_median"]
    )
    shock_audit = {}
    for variant in VARIANTS:
        active = [row for row in rows[variant] if row.get("stress_active")]
        shock_audit[variant] = {
            "active_step_count": len(active),
            "first_zero_based_step": active[0]["step"],
            "last_zero_based_step": active[-1]["step"],
            "effective_learning_rates": sorted(
                {row["effective_learning_rate"] for row in active}
            ),
            "interventions_during_shock": sum(
                bool(row.get("regularized")) for row in active
            ),
            "scheduled_interventions_during_shock": sum(
                bool(row.get("intervention_scheduled")) for row in active
            ),
        }
    integrity = {
        variant: audit_log(logs[variant], results[variant]["completed_steps"])
        for variant in VARIANTS
    }
    analysis = {
        "run": str(RUN.relative_to(ROOT)),
        "source_sha256": manifest["source_sha256"],
        "config": manifest["config"],
        "primary_endpoint": "token accuracy >= 0.99 at three consecutive 125-step validation checkpoints",
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
            "sham_first_target_step": sham["first_target_step"],
            "gsf_first_target_step": gsf["first_target_step"],
            "first_target_delay_steps": gsf["first_target_step"]
            - sham["first_target_step"],
            "sham_confirmed_step": sham["target_confirmed_step"],
            "gsf_confirmed_step": gsf["target_confirmed_step"],
            "confirmed_delay_steps": gsf["target_confirmed_step"]
            - sham["target_confirmed_step"],
            "confirmed_delay_fraction": (
                gsf["target_confirmed_step"] - sham["target_confirmed_step"]
            )
            / sham["target_confirmed_step"],
            "sham_final_validation_loss": sham["evaluation_loss"],
            "gsf_final_validation_loss": gsf["evaluation_loss"],
        },
        "pre_intervention_match": {
            "first_fee_step_zero_based": first_fee_step,
            "compared_steps": first_fee_step,
            "maximum_absolute_task_loss_difference": pre_loss_difference,
            "maximum_absolute_parameter_gradient_norm_difference": pre_gradient_difference,
            "same_first_intervention_schedule": gsf_rows[first_fee_step].get(
                "intervention_scheduled"
            )
            and sham_rows[first_fee_step].get("intervention_scheduled"),
        },
        "gate_audit": {
            "fee_intervention_count": len(interventions),
            "fee_intervention_fraction": len(interventions) / len(gsf_rows),
            "fee_intervention_steps_zero_based": [
                row["step"] for row in interventions
            ],
            "fee_interventions_during_lr_shock": sum(
                row.get("stress_active") is True for row in interventions
            ),
            "sham_scheduled_intervention_count": len(sham_scheduled),
            "sham_scheduled_steps_zero_based": [
                row["step"] for row in sham_scheduled
            ],
        },
        "shock_audit": shock_audit,
        "efficiency_audit": {
            "gs_observer_control": {
                "completed_steps": sham["completed_steps"],
                "timed_training_seconds": sham["timed_training_seconds"],
                "elapsed_seconds": sham["elapsed_seconds"],
                "peak_memory_bytes": sham["peak_memory_bytes"],
                **timing["gs_observer_control"],
            },
            "gs_fe_e_gated": {
                "completed_steps": gsf["completed_steps"],
                "timed_training_seconds": gsf["timed_training_seconds"],
                "elapsed_seconds": gsf["elapsed_seconds"],
                "peak_memory_bytes": gsf["peak_memory_bytes"],
                **timing["gs_fe_e_gated"],
            },
            "observed_training_time_ratio_gsf_over_sham": observed_time_ratio,
            "ordinary_step_system_speed_ratio_gsf_over_sham": ordinary_speed_ratio,
            "system_load_normalized_estimated_time_ratio_gsf_over_sham": normalized_time_ratio,
            "estimated_fee_increment_seconds_per_intervention": fee_increment_per_event,
            "peak_memory_ratio_gsf_over_sham": (
                gsf["peak_memory_bytes"] / sham["peak_memory_bytes"]
            ),
        },
        "interpretation": {
            "result": "GSF reached and confirmed the target 250 updates later than matched GS-SHAM.",
            "causal_strength": "The matched runs agreed closely before the first shared intervention decision at zero-based step 73.",
            "lr_shock": "The 1.5x 32-step shock was applied identically, but no FE-E intervention occurred inside the shock window.",
            "evidence_level": "single-seed paired mechanistic result; not a population estimate",
        },
    }
    return analysis, results


class Plot:
    def __init__(self, width=1600, height=1050, scale=2):
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

    def text(
        self,
        x,
        y,
        value,
        font="label",
        color="#222",
        anchor="la",
        svg_anchor="start",
    ):
        self.draw.text(
            self.pt(x, y),
            str(value),
            font=self.fonts[font],
            fill=color,
            anchor=anchor,
        )
        self.svg.append(
            f'<text class="{font}" x="{x:.2f}" y="{y:.2f}" fill="{color}" text-anchor="{svg_anchor}" dominant-baseline="middle">{escape(str(value))}</text>'
        )

    def line(self, x1, y1, x2, y2, color, width=1, dash=None):
        self.draw.line(
            (*self.pt(x1, y1), *self.pt(x2, y2)),
            fill=color,
            width=max(1, round(width * self.scale)),
        )
        attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.svg.append(
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{color}" stroke-width="{width}"{attr}/>'
        )

    def rect(self, left, top, right, bottom, fill, opacity=1.0):
        self.draw.rectangle(
            (*self.pt(left, top), *self.pt(right, bottom)),
            fill=fill,
        )
        self.svg.append(
            f'<rect x="{left:.2f}" y="{top:.2f}" width="{right-left:.2f}" height="{bottom-top:.2f}" fill="{fill}" opacity="{opacity}"/>'
        )

    def circle(self, x, y, radius, fill):
        px, py = self.pt(x, y)
        rr = round(radius * self.scale)
        self.draw.ellipse((px - rr, py - rr, px + rr, py + rr), fill=fill)
        self.svg.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius}" fill="{fill}"/>'
        )

    def polyline(self, points, color, width=3):
        self.draw.line(
            [self.pt(x, y) for x, y in points],
            fill=color,
            width=round(width * self.scale),
            joint="curve",
        )
        payload = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        self.svg.append(
            f'<polyline points="{payload}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linecap="round" stroke-linejoin="round"/>'
        )

    def save(self):
        FIGURE_BASE.parent.mkdir(parents=True, exist_ok=True)
        self.image.resize(
            (self.width, self.height), Image.Resampling.LANCZOS
        ).save(FIGURE_BASE.with_suffix(".png"), dpi=(180, 180))
        self.svg.append("</svg>")
        FIGURE_BASE.with_suffix(".svg").write_text(
            "\n".join(self.svg), encoding="utf-8"
        )


def make_plot(analysis, results):
    p = Plot()
    p.text(70, 45, "192-layer GS-SHAM versus GSF under sustained LR shock", "title")
    p.text(
        70,
        78,
        "seed 47; width 32; LR 0.002 -> 0.003 at zero-based steps 256-287; target >=99% x3",
        "subtitle",
        "#666",
    )
    for index, key in enumerate(VARIANTS):
        x = 1040 + index * 230
        p.line(x, 74, x + 30, 74, COLORS[key], 4)
        p.circle(x + 15, 74, 4, COLORS[key])
        p.text(x + 40, 74, LABELS[key], "small", "#333")

    x_max = 2500
    interventions = analysis["gate_audit"]["fee_intervention_steps_zero_based"]

    def panel(left, top, right, bottom, metric, y_ticks, transform, title):
        p.text(left, top - 38, title, "panel")
        shock_left = left + 256 / x_max * (right - left)
        shock_right = left + 288 / x_max * (right - left)
        p.rect(shock_left, top, shock_right, bottom, "#FEF3C7", 0.75)
        for value, label in y_ticks:
            y = bottom - transform(value) * (bottom - top)
            p.line(left, y, right, y, "#E6E8EB", 1)
            p.text(left - 12, y, label, "small", "#555", "ra", "end")
        for tick in [0, 500, 1000, 1500, 2000, 2500]:
            x = left + tick / x_max * (right - left)
            p.line(x, top, x, bottom, "#F0F1F3", 1)
            p.text(x, bottom + 22, tick, "small", "#555", "ma", "middle")
        p.line(left, top, left, bottom, "#666", 1.2)
        p.line(left, bottom, right, bottom, "#666", 1.2)
        for step in interventions:
            x = left + (step + 1) / x_max * (right - left)
            p.line(x, bottom - 16, x, bottom, COLORS["gs_fe_e_gated"], 1.8)
        for key in VARIANTS:
            history = results[key]["evaluation_history"]
            points = [
                (
                    left + item["step"] / x_max * (right - left),
                    bottom - transform(item[metric]) * (bottom - top),
                )
                for item in history
            ]
            p.polyline(points, COLORS[key], 3)
            p.circle(points[-1][0], points[-1][1], 4, COLORS[key])
        p.text(
            (left + right) / 2,
            bottom + 55,
            "Optimizer updates",
            "label",
            "#333",
            "ma",
            "middle",
        )

    panel(
        90,
        170,
        760,
        750,
        "evaluation_accuracy",
        [(0, "0.00"), (0.25, "0.25"), (0.5, "0.50"), (0.75, "0.75"), (1, "1.00")],
        lambda value: value,
        "A  Validation token accuracy",
    )
    log_min, log_max = math.log10(0.005), math.log10(4.0)
    panel(
        875,
        170,
        1545,
        750,
        "evaluation_loss",
        [(0.01, "0.01"), (0.1, "0.1"), (1, "1"), (4, "4")],
        lambda value: (math.log10(max(value, 0.005)) - log_min)
        / (log_max - log_min),
        "B  Validation loss (log scale)",
    )
    primary = analysis["primary_comparison"]
    gate = analysis["gate_audit"]
    efficiency = analysis["efficiency_audit"]
    p.text(90, 840, "Matched result", "panel")
    p.text(
        90,
        880,
        f"GS-SHAM: {primary['sham_first_target_step']} first / {primary['sham_confirmed_step']} confirmed",
        "bold",
        COLORS["gs_observer_control"],
    )
    p.text(
        90,
        915,
        f"GSF: {primary['gsf_first_target_step']} first / {primary['gsf_confirmed_step']} confirmed",
        "bold",
        COLORS["gs_fe_e_gated"],
    )
    p.text(
        90,
        950,
        f"GSF delay: {primary['confirmed_delay_steps']} updates ({primary['confirmed_delay_fraction']:.1%}); {gate['fee_intervention_count']} FE-E updates ({gate['fee_intervention_fraction']:.2%}).",
        "label",
        "#333",
    )
    p.text(
        90,
        985,
        f"Observed time ratio {efficiency['observed_training_time_ratio_gsf_over_sham']:.2f}x; system-load-normalized estimate {efficiency['system_load_normalized_estimated_time_ratio_gsf_over_sham']:.2f}x; peak memory {efficiency['peak_memory_ratio_gsf_over_sham']:.2f}x.",
        "subtitle",
        "#555",
    )
    p.text(
        90,
        1017,
        "Pale band: LR shock. Green rug: FE-E updates. No FE-E update occurred inside the shock window.",
        "subtitle",
        "#555",
    )
    p.save()


def write_report(analysis):
    primary = analysis["primary_comparison"]
    match = analysis["pre_intervention_match"]
    gate = analysis["gate_audit"]
    efficiency = analysis["efficiency_audit"]
    sham_eff = efficiency["gs_observer_control"]
    gsf_eff = efficiency["gs_fe_e_gated"]
    lines = [
        "# 192层 seed 47：GS-SHAM 对 GSF（持续学习率冲击）",
        "",
        "## 结论",
        "",
        f"在同构观察器和相同学习率冲击下，GS-SHAM于第 {primary['sham_first_target_step']} 步首次达到 token accuracy ≥99%，第 {primary['sham_confirmed_step']} 步完成连续3次确认；GSF分别为第 {primary['gsf_first_target_step']} 与 {primary['gsf_confirmed_step']} 步。GSF在确认终点上落后 {primary['confirmed_delay_steps']} 步（{primary['confirmed_delay_fraction']:.1%}）。",
        "",
        f"GSF实际介入 {gate['fee_intervention_count']} 次，占其 {gsf_eff['completed_steps']} 次更新的 {gate['fee_intervention_fraction']:.2%}。本轮单种子结果为负：FE-E改变了训练轨迹，但把快速学习相变推迟约250步。",
        "",
        "## 冻结协议",
        "",
        "- seed 47；192层、宽度32、4头、序列长度12、batch size 8。",
        "- 基础学习率0.002；零基第256–287步统一提高到0.003（1.5倍、32步），第288步恢复。",
        "- 两边均使用GS和完全相同的观察器、哨兵探测与状态机；sham只记录计划介入，GSF才注入FE-E梯度。",
        "- 每125步验证一次，每次8个固定batch；token accuracy ≥99%连续3次确认；最多5000步并在确认后停止。",
        "- 当前至少2个传播指标异常、最近4次探测至少3次异常、连续任务伤害确认、单步介入、冷却48步；FE-E梯度目标比率5%→10%→20%。",
        "",
        "## 主结果",
        "",
        "| 方法 | 首次≥99% | 连续确认 | 相对确认步数 | 实际FE-E更新 |",
        "|---|---:|---:|---:|---:|",
        f"| GS-SHAM | {primary['sham_first_target_step']} | {primary['sham_confirmed_step']} | 基准 | 0 |",
        f"| GSF | {primary['gsf_first_target_step']} | {primary['gsf_confirmed_step']} | +{primary['confirmed_delay_steps']}（+{primary['confirmed_delay_fraction']:.1%}） | {gate['fee_intervention_count']}（{gate['fee_intervention_fraction']:.2%}） |",
        "",
        f"GS-SHAM在第1875步验证准确率已达78.91%，第2000步达到100%；GSF在第2125步仍只有10.16%，第2250步才达到100%。这表明差异来自快速相变时点，而不是最终能否拟合任务。",
        "",
        "## 学习率冲击审计",
        "",
        "- 两边均准确记录32个`stress_active=true`步骤，范围为零基第256–287步，有效学习率均为0.003。",
        f"- GSF在冲击窗口内实际介入 {gate['fee_interventions_during_lr_shock']} 次；本轮所有FE-E更新都发生在冲击窗口之外。",
        f"- 首次FE-E更新发生在零基第 {match['first_fee_step_zero_based']} 步，早于学习率冲击。这说明192层seed 47在自然训练阶段已经满足当前门控条件。",
        "- 学习率冲击没有触发即时FE-E救援，因此本轮不能把相变差异解释为FE-E直接修复了冲击。",
        "",
        "## 同构与因果审计",
        "",
        f"- 首次FE-E更新前比较 {match['compared_steps']} 个完整步骤，最大任务损失差仅 {match['maximum_absolute_task_loss_difference']:.2e}，最大参数梯度范数差 {match['maximum_absolute_parameter_gradient_norm_difference']:.2e}。",
        f"- 零基第 {match['first_fee_step_zero_based']} 步两边都计划介入；GS-SHAM推进相同状态机但不应用FE-E，GSF应用FE-E。首次分叉因而具有良好的同构对照。",
        f"- GSF介入步：{gate['fee_intervention_steps_zero_based']}。",
        f"- GS-SHAM计划性sham步：{gate['sham_scheduled_steps_zero_based']}。首次分叉后的介入时点不同是参数轨迹和观察状态已经改变的正常结果。",
        "",
        "## 计算代价",
        "",
        f"- 实测训练时间：GS-SHAM {sham_eff['timed_training_seconds']:.1f}s，GSF {gsf_eff['timed_training_seconds']:.1f}s，GSF高 {efficiency['observed_training_time_ratio_gsf_over_sham']-1:.1%}。",
        f"- 后运行的GSF普通步中位数为 {gsf_eff['ordinary_step_seconds_median']:.3f}s，GS-SHAM为 {sham_eff['ordinary_step_seconds_median']:.3f}s，说明串行运行期间系统整体慢了约 {efficiency['ordinary_step_system_speed_ratio_gsf_over_sham']-1:.1%}。按普通步速度归一化后，GSF达到终点的估计总计算成本仍约高 {efficiency['system_load_normalized_estimated_time_ratio_gsf_over_sham']-1:.1%}。",
        f"- GSF观察器探测步中位数 {gsf_eff['observer_probe_step_seconds_median']:.3f}s，FE-E介入步 {gsf_eff['fee_intervention_step_seconds_median']:.3f}s；每次真实介入相对同类探测约多 {efficiency['estimated_fee_increment_seconds_per_intervention']:.3f}s。",
        f"- 峰值内存：GS-SHAM {sham_eff['peak_memory_bytes']/1e6:.1f}MB，GSF {gsf_eff['peak_memory_bytes']/1e6:.1f}MB，约 {efficiency['peak_memory_ratio_gsf_over_sham']:.2f}倍。",
        "",
        "## 研究含义",
        "",
        "1. 本轮不支持“传播越深，当前门控FE-E价值必然越高”的单调命题；192层下当前门控产生了明显负向相变延迟。",
        "2. 结果更像门控适应证错误，而不是FE-E完全没有作用：一次或多次FE-E更新足以改变长期轨迹，但改变方向不利。",
        "3. 观察器在冲击前就判定自然传播有害，说明当前异常分数会把某些可自行收敛的深层状态识别为需要治疗。下一步应研究治疗收益判断，而不是单纯降低传播异常阈值。",
        "4. 只有一个seed，不能估计总体效果；但同构首分叉很干净，足以作为一个可信的负机制案例保留。",
        "",
        "## 文件与完整性",
        "",
        f"- 原始运行：`{analysis['run']}`",
        "- 结构化分析：`results/mlx_d192_s47_acc99_gs_sham_vs_gsf_lrshock_analysis.json`",
        "- 图表：`output/figures/fee_d192_s47_gs_sham_vs_gsf_lrshock.png`与`.svg`",
        "- 分析脚本：`scripts/analyze_mlx_d192_s47_gs_sham_vs_gsf_lrshock.py`",
        f"- 源码SHA-256：`{analysis['source_sha256']}`",
        "- 两条正式日志均有连续零基步骤、完成记录，所有步骤均为有限数值。遗漏学习率冲击的16步预运行已移入不随仓库分发的本地审计归档；运行标识和失效原因保留，但不纳入结果。",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    analysis, results = analyze()
    ANALYSIS_PATH.parent.mkdir(parents=True, exist_ok=True)
    ANALYSIS_PATH.write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_report(analysis)
    make_plot(analysis, results)
    print(ANALYSIS_PATH)
    print(REPORT_PATH)
    print(FIGURE_BASE.with_suffix(".png"))
    print(FIGURE_BASE.with_suffix(".svg"))


if __name__ == "__main__":
    main()
