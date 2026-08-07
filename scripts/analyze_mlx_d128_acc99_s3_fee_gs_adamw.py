#!/usr/bin/env python3
"""Aggregate the three completed 128-layer FE-E/GS/AdamW seed runs."""

from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
import statistics
from xml.sax.saxutils import escape

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    31: ROOT / "results/mlx_d128_s4_acc99_fee_gs_adamw_hybrid_first/run_20260805_163941",
    47: ROOT / "results/mlx_d128_s47_acc99_fee_gs_adamw_hybrid_first/run_20260805_161221",
    59: ROOT / "results/mlx_d128_s4_acc99_fee_gs_adamw_hybrid_first/run_20260805_163941",
}
PARTIAL_LOG = RUNS[31] / "logs/gs_fe_e_gated_seed71.jsonl"
MANUAL_STOP_AUDIT = RUNS[31] / "manual_stop_audit.json"
VARIANTS = ["gs_fe_e_gated", "gradient_smoothing", "baseline"]
LABELS = {
    "gs_fe_e_gated": "GS + gated FE-E",
    "gradient_smoothing": "Pure GS",
    "baseline": "AdamW",
}
COLORS = {
    "gs_fe_e_gated": "#0B6E69",
    "gradient_smoothing": "#D97706",
    "baseline": "#2563A6",
}
SUMMARY_PATH = ROOT / "results/mlx_d128_acc99_s3_fee_gs_adamw_analysis.json"
REPORT_PATH = ROOT / "docs/mlx_d128_acc99_s3_fee_gs_adamw_report.md"
FIGURE_BASE = ROOT / "output/figures/fee_d128_acc99_s3_fee_gs_adamw"
TAU = 5000


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def mean(values):
    return statistics.fmean(values)


def sample_sd(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0


def fmt_step(value):
    return ">5000" if value is None else str(value)


def load_data():
    results = {}
    logs = {}
    for seed, run in RUNS.items():
        for variant in VARIANTS:
            key = (seed, variant)
            results[key] = read_json(run / "runs" / f"{variant}_seed{seed}.json")
            logs[key] = read_jsonl(run / "logs" / f"{variant}_seed{seed}.jsonl")
    return results, logs


def audit_log(records, expected_steps):
    steps = [row for row in records if row.get("record_type") == "train_step"]
    starts = [row for row in records if row.get("record_type") == "run_start"]
    ends = [row for row in records if row.get("record_type") == "run_end"]
    return {
        "records": len(records),
        "train_steps": len(steps),
        "expected_steps": expected_steps,
        "run_start_records": len(starts),
        "run_end_records": len(ends),
        "contiguous_zero_based_steps": [row["step"] for row in steps]
        == list(range(expected_steps)),
        "all_steps_finite": all(row.get("finite") is True for row in steps),
        "completed_run_end": len(ends) == 1 and ends[0].get("status") == "completed",
    }


def compare(left, right, results):
    wins = ties = losses = 0
    per_seed = {}
    for seed in RUNS:
        a = results[(seed, left)]["target_confirmed_step"]
        b = results[(seed, right)]["target_confirmed_step"]
        ar = TAU if a is None else a
        br = TAU if b is None else b
        if a is not None and b is None:
            outcome = "win"
        elif a is None and b is not None:
            outcome = "loss"
        elif ar < br:
            outcome = "win"
        elif ar > br:
            outcome = "loss"
        else:
            outcome = "tie"
        wins += outcome == "win"
        ties += outcome == "tie"
        losses += outcome == "loss"
        per_seed[str(seed)] = {
            "left_confirmed_step": a,
            "right_confirmed_step": b,
            "restricted_difference_left_minus_right": ar - br,
            "outcome_for_left": outcome,
        }
    return {"wins": wins, "ties": ties, "losses": losses, "per_seed": per_seed}


def analyze(results, logs):
    source_hashes = sorted(
        {
            read_json(run / "manifest.json")["source_sha256"]
            for run in set(RUNS.values())
        }
    )
    configs = [
        next(row for row in logs[(seed, "gs_fe_e_gated")] if row["record_type"] == "run_start")["config"]
        for seed in RUNS
    ]
    integrity = {
        f"{variant}_seed{seed}": audit_log(
            logs[(seed, variant)], results[(seed, variant)]["completed_steps"]
        )
        for seed in RUNS
        for variant in VARIANTS
    }
    method_summary = {}
    for variant in VARIANTS:
        confirmed = [results[(seed, variant)]["target_confirmed_step"] for seed in RUNS]
        first = [results[(seed, variant)]["first_target_step"] for seed in RUNS]
        restricted_confirmed = [TAU if value is None else value for value in confirmed]
        restricted_first = [TAU if value is None else value for value in first]
        method_summary[variant] = {
            "label": LABELS[variant],
            "confirmed_steps_by_seed": dict(zip(map(str, RUNS), confirmed)),
            "first_target_steps_by_seed": dict(zip(map(str, RUNS), first)),
            "completion_count": sum(value is not None for value in confirmed),
            "completion_rate": sum(value is not None for value in confirmed) / len(confirmed),
            "restricted_confirmed_mean": mean(restricted_confirmed),
            "restricted_confirmed_sample_sd": sample_sd(restricted_confirmed),
            "restricted_confirmed_median": statistics.median(restricted_confirmed),
            "restricted_first_mean": mean(restricted_first),
            "mean_step_seconds": mean(
                [results[(seed, variant)]["mean_step_seconds"] for seed in RUNS]
            ),
            "peak_memory_bytes_max": max(
                results[(seed, variant)]["peak_memory_bytes"] for seed in RUNS
            ),
        }

    gate_by_seed = {}
    total_hybrid_steps = 0
    total_interventions = 0
    for seed in RUNS:
        train = [
            row
            for row in logs[(seed, "gs_fe_e_gated")]
            if row.get("record_type") == "train_step"
        ]
        interventions = [row for row in train if row.get("regularized")]
        probes = [row for row in train if row.get("observer_probe")]
        total_hybrid_steps += len(train)
        total_interventions += len(interventions)
        gate_by_seed[str(seed)] = {
            "training_steps": len(train),
            "observer_probes": len(probes),
            "intervention_count": len(interventions),
            "intervention_fraction": len(interventions) / len(train),
            "intervention_steps_zero_based": [row["step"] for row in interventions],
            "gradient_ratio_counts": dict(
                sorted(Counter(str(round(row["observer_gradient_ratio"], 3)) for row in interventions).items())
            ),
            "confirmed_harm_events": sum(
                bool(row.get("observer_confirmed_harm")) for row in train
            ),
        }

    seed31_hybrid = [
        row
        for row in logs[(31, "gs_fe_e_gated")]
        if row.get("record_type") == "train_step"
    ]
    seed31_gs = [
        row
        for row in logs[(31, "gradient_smoothing")]
        if row.get("record_type") == "train_step"
    ]
    first_fee = gate_by_seed["31"]["intervention_steps_zero_based"][0]
    pre_fee_step = 50
    pre_fee_divergence = {
        "first_fee_step_zero_based": first_fee,
        "check_step_zero_based": pre_fee_step,
        "hybrid_task_loss": seed31_hybrid[pre_fee_step]["task_loss"],
        "pure_gs_task_loss": seed31_gs[pre_fee_step]["task_loss"],
        "hybrid_parameter_gradient_norm": seed31_hybrid[pre_fee_step]["parameter_gradient_norm"],
        "pure_gs_parameter_gradient_norm": seed31_gs[pre_fee_step]["parameter_gradient_norm"],
    }

    partial = read_jsonl(PARTIAL_LOG) if PARTIAL_LOG.exists() else []
    partial_steps = [row for row in partial if row.get("record_type") == "train_step"]
    stop_audit = read_json(MANUAL_STOP_AUDIT) if MANUAL_STOP_AUDIT.exists() else {}
    excluded_partial = stop_audit.get("excluded_partial_run", {})
    partial_step_count = len(partial_steps) or int(excluded_partial.get("logged_train_steps", 0))
    partial_run_end = (
        any(row.get("record_type") == "run_end" for row in partial)
        if partial
        else bool(excluded_partial.get("run_end_record_present", False))
    )
    return {
        "analysis_scope": {
            "completed_seeds": list(RUNS),
            "planned_seeds": [31, 47, 59, 71, 89],
            "stopped_early_at_user_request": True,
            "excluded_partial_run": {
                "seed": 71,
                "variant": "gs_fe_e_gated",
                "train_steps_logged": partial_step_count,
                "run_end_present": partial_run_end,
            },
        },
        "primary_endpoint": "token accuracy >= 0.99 for 3 consecutive 125-step validation checkpoints",
        "censoring_horizon_updates": TAU,
        "source_sha256_values": source_hashes,
        "identical_source_hash_across_runs": len(source_hashes) == 1,
        "identical_config_across_completed_seeds": all(config == configs[0] for config in configs[1:]),
        "config": configs[0],
        "integrity": integrity,
        "all_completed_logs_pass_integrity": all(
            all(
                item[key]
                for key in [
                    "contiguous_zero_based_steps",
                    "all_steps_finite",
                    "completed_run_end",
                ]
            )
            and item["run_start_records"] == 1
            and item["run_end_records"] == 1
            for item in integrity.values()
        ),
        "methods": method_summary,
        "pairwise": {
            "hybrid_vs_pure_gs": compare("gs_fe_e_gated", "gradient_smoothing", results),
            "hybrid_vs_adamw": compare("gs_fe_e_gated", "baseline", results),
            "pure_gs_vs_adamw": compare("gradient_smoothing", "baseline", results),
        },
        "gate_audit": {
            "by_seed": gate_by_seed,
            "total_interventions": total_interventions,
            "total_hybrid_steps": total_hybrid_steps,
            "aggregate_intervention_fraction": total_interventions / total_hybrid_steps,
        },
        "pre_intervention_causal_check_seed31": pre_fee_divergence,
        "interpretation": {
            "supported": [
                "The gated hybrid completed all three seeds while pure GS failed within 5000 updates on seed 31.",
                "The gate remained dormant on seeds 47 and 59, where the hybrid tied pure GS exactly at the endpoint.",
                "FE-E therefore shows a reliability-rescue signal for GS in this small synthetic stress test.",
            ],
            "not_supported": [
                "Statistical superiority from n=3.",
                "Faster convergence than AdamW: AdamW had lower restricted mean steps and beat the hybrid on two of three seeds.",
                "A clean causal attribution of seed 31 rescue to FE-E because trajectories diverged before the first FE-E update.",
                "Wall-clock speedup under serial, load-contaminated execution.",
            ],
        },
    }


class Plotter:
    def __init__(self, width=1800, height=1420, scale=2):
        self.width = width
        self.height = height
        self.scale = scale
        self.image = Image.new("RGB", (width * scale, height * scale), "white")
        self.draw = ImageDraw.Draw(self.image, "RGBA")
        regular = "/System/Library/Fonts/Supplemental/Arial.ttf"
        bold = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
        self.fonts = {
            "title": ImageFont.truetype(bold, 27 * scale),
            "subtitle": ImageFont.truetype(regular, 16 * scale),
            "panel": ImageFont.truetype(bold, 18 * scale),
            "label": ImageFont.truetype(regular, 15 * scale),
            "small": ImageFont.truetype(regular, 13 * scale),
            "bold": ImageFont.truetype(bold, 14 * scale),
        }
        self.svg = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="#fff"/>',
            '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#222}.title{font-size:27px;font-weight:700}.subtitle{font-size:16px;fill:#666}.panel{font-size:18px;font-weight:700}.label{font-size:15px}.small{font-size:13px;fill:#555}.bold{font-size:14px;font-weight:700}</style>',
        ]

    def pt(self, x, y):
        return round(x * self.scale), round(y * self.scale)

    def line(self, x1, y1, x2, y2, color, width=1, dash=None):
        self.draw.line((*self.pt(x1, y1), *self.pt(x2, y2)), fill=color, width=max(1, round(width * self.scale)))
        attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.svg.append(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{color}" stroke-width="{width}"{attr}/>' )

    def text(self, x, y, value, font="label", color="#222", anchor="la", svg_anchor="start"):
        self.draw.text(self.pt(x, y), value, font=self.fonts[font], fill=color, anchor=anchor)
        cls = font
        self.svg.append(f'<text class="{cls}" x="{x:.2f}" y="{y:.2f}" fill="{color}" text-anchor="{svg_anchor}" dominant-baseline="middle">{escape(str(value))}</text>')

    def circle(self, x, y, radius, fill, outline=None, width=1):
        px, py = self.pt(x, y)
        rr = round(radius * self.scale)
        self.draw.ellipse((px-rr, py-rr, px+rr, py+rr), fill=fill, outline=outline, width=max(1, round(width*self.scale)))
        self.svg.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius}" fill="{fill}" stroke="{outline or fill}" stroke-width="{width}"/>')

    def rect(self, x1, y1, x2, y2, fill, outline=None, width=1):
        self.draw.rectangle((*self.pt(x1, y1), *self.pt(x2, y2)), fill=fill, outline=outline, width=max(1, round(width*self.scale)))
        self.svg.append(f'<rect x="{x1:.2f}" y="{y1:.2f}" width="{x2-x1:.2f}" height="{y2-y1:.2f}" fill="{fill}" stroke="{outline or fill}" stroke-width="{width}"/>')

    def polyline(self, points, color, width=3):
        self.draw.line([self.pt(x, y) for x, y in points], fill=color, width=round(width*self.scale), joint="curve")
        payload = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        self.svg.append(f'<polyline points="{payload}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linecap="round" stroke-linejoin="round"/>')

    def save(self):
        FIGURE_BASE.parent.mkdir(parents=True, exist_ok=True)
        self.image.resize((self.width, self.height), Image.Resampling.LANCZOS).save(FIGURE_BASE.with_suffix(".png"), dpi=(180, 180))
        self.svg.append("</svg>")
        FIGURE_BASE.with_suffix(".svg").write_text("\n".join(self.svg), encoding="utf-8")


def plot(results, analysis):
    p = Plotter()
    p.text(70, 48, "128-layer FE-E stress test: three completed seeds", "title")
    p.text(70, 82, "Primary endpoint: token accuracy >=99% at 3 consecutive checkpoints; lower updates are better", "subtitle", "#666")
    legend_x = 1040
    for index, variant in enumerate(VARIANTS):
        x = legend_x + index * 225
        p.line(x, 76, x + 30, 76, COLORS[variant], 4)
        p.circle(x + 15, 76, 4, COLORS[variant])
        p.text(x + 40, 76, LABELS[variant], "small", "#333")

    # Panel A: validation accuracy curves, one facet per seed.
    p.text(70, 132, "A  Validation accuracy trajectories", "panel")
    plot_left, plot_right = 90, 1725
    gap = 42
    facet_w = (plot_right - plot_left - 2 * gap) / 3
    top, bottom = 180, 650
    for facet, seed in enumerate(RUNS):
        left = plot_left + facet * (facet_w + gap)
        right = left + facet_w
        for tick in [0, .25, .5, .75, 1.0]:
            y = bottom - tick * (bottom - top)
            p.line(left, y, right, y, "#E6E8EB", 1)
            if facet == 0:
                p.text(left - 12, y, f"{tick:.2f}", "small", "#555", "ra", "end")
        for tick in [0, 1000, 2000, 3000, 4000, 5000]:
            x = left + tick / TAU * (right - left)
            p.line(x, top, x, bottom, "#F0F1F3", 1)
            p.text(x, bottom + 22, str(tick), "small", "#555", "ma", "middle")
        y99 = bottom - .99 * (bottom - top)
        p.line(left, y99, right, y99, "#7A7A7A", 1.5, "6 5")
        p.text((left + right) / 2, top - 23, f"Seed {seed}", "bold", "#222", "ma", "middle")
        p.line(left, top, left, bottom, "#666", 1.3)
        p.line(left, bottom, right, bottom, "#666", 1.3)
        for variant in VARIANTS:
            history = results[(seed, variant)]["evaluation_history"]
            points = [
                (
                    left + item["step"] / TAU * (right - left),
                    bottom - item["evaluation_accuracy"] * (bottom - top),
                )
                for item in history
            ]
            if points:
                p.polyline(points, COLORS[variant], 3)
                p.circle(points[-1][0], points[-1][1], 4.5, COLORS[variant], "white", 1)
    p.text((plot_left + plot_right) / 2, bottom + 58, "Optimizer updates", "label", "#333", "ma", "middle")

    # Panel B: per-seed confirmation steps including censoring.
    p.text(70, 745, "B  Updates to confirmed endpoint", "panel")
    left, right, top, bottom = 245, 1130, 795, 1175
    for tick in [0, 1000, 2000, 3000, 4000, 5000]:
        x = left + tick / TAU * (right - left)
        p.line(x, top, x, bottom, "#E6E8EB", 1)
        p.text(x, bottom + 22, str(tick), "small", "#555", "ma", "middle")
    offsets = {"gs_fe_e_gated": -25, "gradient_smoothing": 0, "baseline": 25}
    for row, seed in enumerate(RUNS):
        y0 = top + 73 + row * 112
        p.text(left - 38, y0, f"Seed {seed}", "bold", "#333", "ra", "end")
        for variant in VARIANTS:
            value = results[(seed, variant)]["target_confirmed_step"]
            y = y0 + offsets[variant]
            if value is None:
                x = right
                p.line(x - 7, y - 7, x + 7, y + 7, COLORS[variant], 3)
                p.line(x - 7, y + 7, x + 7, y - 7, COLORS[variant], 3)
                p.text(x - 12, y, ">5000", "small", "#333", "ra", "end")
            else:
                x = left + value / TAU * (right - left)
                p.circle(x, y, 7, COLORS[variant], "white", 1)
                p.text(x + 12, y, str(value), "small", "#333")
    p.line(left, bottom, right, bottom, "#666", 1.3)
    p.text((left + right) / 2, bottom + 58, "Updates (censoring horizon = 5000)", "label", "#333", "ma", "middle")

    # Panel C: intervention count and restricted mean summary.
    p.text(1240, 745, "C  Gate activity and restricted mean", "panel")
    bar_left, bar_right = 1270, 1715
    bar_top, bar_bottom = 810, 1000
    counts = [analysis["gate_audit"]["by_seed"][str(seed)]["intervention_count"] for seed in RUNS]
    max_count = max(counts) or 1
    for index, (seed, count) in enumerate(zip(RUNS, counts)):
        x1 = bar_left + 30 + index * 142
        x2 = x1 + 72
        y = bar_bottom - count / max_count * (bar_bottom - bar_top - 25)
        p.rect(x1, y, x2, bar_bottom, COLORS["gs_fe_e_gated"], None, 0)
        p.text((x1 + x2) / 2, y - 16, str(count), "bold", "#222", "ma", "middle")
        p.text((x1 + x2) / 2, bar_bottom + 22, f"Seed {seed}", "small", "#555", "ma", "middle")
    p.line(bar_left, bar_bottom, bar_right, bar_bottom, "#666", 1.2)
    p.text(bar_left, 1048, "Restricted mean updates to confirmation", "bold", "#333")
    y = 1080
    for variant in VARIANTS:
        value = analysis["methods"][variant]["restricted_confirmed_mean"]
        completion = analysis["methods"][variant]["completion_count"]
        p.circle(bar_left + 8, y, 5, COLORS[variant])
        p.text(bar_left + 23, y, f"{LABELS[variant]}: {value:.0f}  ({completion}/3 completed)", "small", "#333")
        y += 34
    p.text(70, 1330, "Censored pure-GS seed 31 is retained at the 5000-update horizon. Wall-clock time is excluded from ranking because runs were serial and system load changed.", "subtitle", "#555")
    p.text(70, 1362, "Evidence level: engineering signal only (n=3, synthetic reverse-sequence task); AdamW remains faster on restricted mean updates.", "subtitle", "#555")
    p.save()


def write_report(analysis):
    methods = analysis["methods"]
    pair_gs = analysis["pairwise"]["hybrid_vs_pure_gs"]
    pair_adam = analysis["pairwise"]["hybrid_vs_adamw"]
    lines = [
        "# 128 层 FE-E / GS / AdamW 三种子提前停止报告",
        "",
        "## 结论",
        "",
        f"按用户要求，原计划 5 种子在完成 seed 59 后停止。正式统计包含 seed 31、47、59；seed 71 在停止信号生效前记录了 {analysis['analysis_scope']['excluded_partial_run']['train_steps_logged']} 步，未完成且不纳入结果，seed 89 未启动。",
        "",
        "本轮最强的正向信号不是 FE-E 平均更快，而是**门控 FE-E 提高了纯 GS 的压力可靠性**：混合方案 3/3 达标，纯 GS 2/3 达标。seed 31 中纯 GS 到 5000 步仍未达标，而混合方案在 3625 步确认；seed 47 和 59 中门控均 0 次介入，混合方案与纯 GS 的确认步数完全相同。",
        "",
        "但 FE-E 尚未表现出相对 AdamW 的总体优势：AdamW 3/3 达标，5000 步删失上限下的受限平均确认步数为 2083，优于混合方案的 2625；混合方案相对 AdamW 为 1 胜、0 平、2 负。因此当前结论应是“FE-E 可能是 GS 的按需失稳恢复器”，而不是“FE-E 已优于常规 AdamW”。",
        "",
        "## 冻结协议",
        "",
        "- 128 层、宽度 32、4 头、序列长度 12、batch size 8，学习率 0.002。",
        "- 每 125 步验证一次，每次 8 个固定验证 batch。",
        "- 主终点：token accuracy ≥99%，连续 3 个检查点成立；最多 5000 更新。",
        "- 每个种子固定先混合方案，再纯 GS，最后 AdamW。",
        "- FE-E 观察器每 8 步探测；24 步校准；当前至少 2 个指标异常；最近 4 次探测至少 3 次异常；任务哨兵连续受损 2 次；单步介入；冷却 48 步；梯度比率 5%→10%→20%。",
        "",
        "## 主结果",
        "",
        "| 种子 | GS + 门控 FE-E | 纯 GS | AdamW | FE-E 介入 |",
        "|---:|---:|---:|---:|---:|",
    ]
    results, _ = load_data()
    for seed in RUNS:
        lines.append(
            f"| {seed} | {fmt_step(results[(seed, 'gs_fe_e_gated')]['target_confirmed_step'])} | "
            f"{fmt_step(results[(seed, 'gradient_smoothing')]['target_confirmed_step'])} | "
            f"{fmt_step(results[(seed, 'baseline')]['target_confirmed_step'])} | "
            f"{analysis['gate_audit']['by_seed'][str(seed)]['intervention_count']} |"
        )
    lines += [
        "",
        "`>5000` 表示在预设上限内未确认终点，不是把失败样本删除。",
        "",
        "| 方法 | 达标率 | 受限平均确认步数（τ=5000） | 样本标准差 | 中位数 |",
        "|---|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        item = methods[variant]
        lines.append(
            f"| {LABELS[variant]} | {item['completion_count']}/3 | "
            f"{item['restricted_confirmed_mean']:.0f} | {item['restricted_confirmed_sample_sd']:.0f} | "
            f"{item['restricted_confirmed_median']:.0f} |"
        )
    saving = methods["gradient_smoothing"]["restricted_confirmed_mean"] - methods["gs_fe_e_gated"]["restricted_confirmed_mean"]
    frac = saving / methods["gradient_smoothing"]["restricted_confirmed_mean"]
    lines += [
        "",
        f"相对纯 GS，混合方案的受限平均确认步数减少 {saving:.0f} 次更新（{frac:.1%}），配对结果为 {pair_gs['wins']} 胜 / {pair_gs['ties']} 平 / {pair_gs['losses']} 负。该数字由一个删失失败种子主导，n=3 不足以做显著性主张。",
        "",
        f"相对 AdamW，混合方案为 {pair_adam['wins']} 胜 / {pair_adam['ties']} 平 / {pair_adam['losses']} 负；受限平均多用 {methods['gs_fe_e_gated']['restricted_confirmed_mean'] - methods['baseline']['restricted_confirmed_mean']:.0f} 次更新。",
        "",
        "## 门控审计",
        "",
        f"- 三个混合 run 共 {analysis['gate_audit']['total_hybrid_steps']} 步，FE-E 实际介入 {analysis['gate_audit']['total_interventions']} 步，占 {analysis['gate_audit']['aggregate_intervention_fraction']:.3%}。",
        "- 23 次介入全部发生在 seed 31；seed 47、59 均为 0 次。后两者的混合方案与纯 GS 终点严格平局，支持门控在无持续伤害时保持休眠。",
        f"- 活跃 seed 31 的 FE-E 更新占比为 {analysis['gate_audit']['by_seed']['31']['intervention_fraction']:.3%}，首个介入为零基第 {analysis['pre_intervention_causal_check_seed31']['first_fee_step_zero_based']} 步。",
        f"- 活跃混合 run 的峰值内存为 {methods['gs_fe_e_gated']['peak_memory_bytes_max']/1e6:.1f} MB；纯 GS 最大为 {methods['gradient_smoothing']['peak_memory_bytes_max']/1e6:.1f} MB。该微型模型上的倍数不能直接外推到 7B，但说明二阶探测/正则会增加工作集。",
        "",
        "## 因果与证据边界",
        "",
        "1. seed 31 的混合与纯 GS 在首次 FE-E 介入前已发生数值轨迹分离：首个 FE-E 更新是零基第 177 步，但第 50 步参数梯度范数已分别为 168.13 与 23.03。Metal 深层训练对微小数值顺序敏感，因此不能把 seed 31 的全部救援效果干净归因于 FE-E。",
        "2. n=3 且是合成反序列任务；没有统计功效证明总体优越性，也不能外推到 7B 语言模型。",
        "3. 固定串行顺序与系统后台负载污染墙钟；本报告只按 updates-to-target 排名，时间仅保留为审计字段。",
        "4. 本轮缺少同计算图的 sham observer 控制。下一轮最有价值的实验不是盲目扩大种子，而是加入“只探测、不介入”的观察器对照，并在 CUDA/确定性更好的环境复现 seed 31。",
        "5. 由于用户要求提前停止，seed 71/89 不属于缺失随机样本意义上的完整五种子结果；本报告明确标为三种子探索性结果。",
        "",
        "## 工程意义",
        "",
        "当前证据支持一种窄而有用的定位：FE-E 不是常开优化器，也不是 AdamW 替代品，而是 GS 训练中的按需保护层。它在两个正常种子中不介入，在一个困难种子中介入并使训练达到终点。若这种“失败率下降、正常轨迹不受扰动”的性质能在真实语言建模和更多种子上复现，生产价值主要体现为减少深层训练作业失败与重跑，而不是单步更快。",
        "",
        "## 文件与完整性",
        "",
        "- 原始 seed 47：`results/mlx_d128_s47_acc99_fee_gs_adamw_hybrid_first/run_20260805_161221`",
        "- 原始 seed 31/59：`results/mlx_d128_s4_acc99_fee_gs_adamw_hybrid_first/run_20260805_163941`",
        "- 主动停止审计：`results/mlx_d128_s4_acc99_fee_gs_adamw_hybrid_first/run_20260805_163941/manual_stop_audit.json`",
        "- 结构化分析：`results/mlx_d128_acc99_s3_fee_gs_adamw_analysis.json`",
        "- 图表：`output/figures/fee_d128_acc99_s3_fee_gs_adamw.png` 与 `.svg`",
        f"- 源码 SHA-256：`{analysis['source_sha256_values'][0]}`；两个运行目录哈希一致。",
        "- 9 个正式 run 均有 1 条 run_start、连续逐步 train_step、1 条 completed run_end，且无 NaN/Inf。seed 71 的不完整日志单独保留，不纳入统计。",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    results, logs = load_data()
    analysis = analyze(results, logs)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(analysis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(analysis)
    plot(results, analysis)
    print(SUMMARY_PATH)
    print(REPORT_PATH)
    print(FIGURE_BASE.with_suffix('.png'))
    print(FIGURE_BASE.with_suffix('.svg'))


if __name__ == "__main__":
    main()
