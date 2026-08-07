#!/usr/bin/env python3
"""Audit the intentionally stopped 128-layer, three-case, 5000-step run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
from typing import Any


METHODS = ("gradient_smoothing", "fe_e_always", "gs_fe_e_gated")
LABELS = {
    "gradient_smoothing": "纯 Gradient Smoothing",
    "fe_e_always": "常开 FE-E",
    "gs_fe_e_gated": "GS + 门控 FE-E",
}
THRESHOLDS = (1.0, 0.1, 0.01, 0.001)


def read_log(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    steps = [record for record in records if record["record_type"] == "train_step"]
    return records, steps


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * q)]


def audit(run: Path) -> dict[str, Any]:
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    methods: dict[str, Any] = {}
    for method in METHODS:
        records, steps = read_log(run / "logs" / f"{method}_seed31.jsonl")
        evaluations = [row for row in steps if row["evaluation_loss"] is not None]
        ordinary = [
            float(row["step_seconds"])
            for row in steps
            if not row["regularized"]
            and not row["adjoint_probe"]
            and row["evaluation_loss"] is None
        ]
        timed = [float(row["step_seconds"]) for row in steps[5:]]
        hits = {}
        for threshold in THRESHOLDS:
            hit = next(
                (row for row in evaluations if float(row["evaluation_loss"]) <= threshold),
                None,
            )
            hits[str(threshold)] = None if hit is None else {
                "step": int(hit["step"]) + 1,
                "loss": float(hit["evaluation_loss"]),
                "timed_training_seconds": float(hit["timed_training_seconds"]),
            }
        methods[method] = {
            "train_steps": len(steps),
            "last_step": int(steps[-1]["step"]) + 1,
            "evaluation_points": len(evaluations),
            "last_evaluation_step": int(evaluations[-1]["step"]) + 1,
            "last_evaluation_loss": float(evaluations[-1]["evaluation_loss"]),
            "completed_run_end": any(
                row["record_type"] == "run_end" and row.get("status") == "completed"
                for row in records
            ),
            "nonfinite": sum(row.get("finite") is False for row in steps),
            "threshold_hits": hits,
            "timing": {
                "mean_step_seconds": statistics.fmean(timed),
                "median_step_seconds": statistics.median(timed),
                "p95_step_seconds": percentile(timed, 0.95),
                "ordinary_first500_median": statistics.median(ordinary[:500]) if ordinary else None,
                "ordinary_last500_median": statistics.median(ordinary[-500:]) if ordinary else None,
            },
            "evaluation_history": [
                {
                    "step": int(row["step"]) + 1,
                    "loss": float(row["evaluation_loss"]),
                    "timed_training_seconds": float(row["timed_training_seconds"]),
                }
                for row in evaluations
            ],
        }

    shared_step = min(item["last_evaluation_step"] for item in methods.values())
    shared = {}
    for method, item in methods.items():
        point = next(row for row in item["evaluation_history"] if row["step"] == shared_step)
        shared[method] = point["loss"]
    return {
        "run": str(run),
        "config": manifest["config"],
        "source_sha256": manifest["source_sha256"],
        "stop": {
            "intentional": True,
            "reason": "System load and thermal state changed across sequential cases, invalidating wall-clock comparison.",
            "stopped_method": "gs_fe_e_gated",
            "last_logged_step": methods["gs_fe_e_gated"]["last_step"],
            "requested_threshold": 0.001,
            "threshold_reached_before_stop": False,
        },
        "methods": methods,
        "shared_checkpoint": {"step": shared_step, "losses": shared},
        "system_observation": {
            "pmset_thermal_warning": "none recorded",
            "pmset_performance_warning": "none recorded",
            "concurrent_loads": ["mediaanalysisd", "mds_stores", "Google Chrome Renderer", "OrbStack"],
            "timing_valid_for_algorithm_ranking": False,
        },
    }


def build_report(summary: dict[str, Any]) -> str:
    methods = summary["methods"]
    shared = summary["shared_checkpoint"]
    rows = [
        "| 方法 | 实际步数 | 最后验证损失 | ≤1.0 | ≤0.1 | ≤0.01 | ≤0.001 | 完整结束 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        item = methods[method]
        hits = item["threshold_hits"]
        values = []
        for threshold in THRESHOLDS:
            hit = hits[str(threshold)]
            values.append("未达到" if hit is None else f"{hit['step']} 步")
        rows.append(
            f"| {LABELS[method]} | {item['train_steps']} | {item['last_evaluation_loss']:.6f} | "
            + " | ".join(values)
            + f" | {'是' if item['completed_run_end'] else '否（主动停止）'} |"
        )

    gs = methods["gradient_smoothing"]
    hybrid = methods["gs_fe_e_gated"]
    lines = [
        "# MLX 128 层三方案 5000 步实验主动停止审计",
        "",
        "## 状态",
        "",
        "实验按用户判断主动停止。原因不是模型出现 NaN/Inf，而是三个方案串行运行期间系统负载和热状态发生变化，后运行方案的非介入步也明显慢于早期纯 GS，墙钟时间不再满足可比性。",
        "",
        f"纯 GS 和常开 FE-E 各完整运行 5000 步；GS + 门控 FE-E 在第 {hybrid['train_steps']} 步后停止，最后一个验证检查点是第 {hybrid['last_evaluation_step']} 步。原始日志不补写伪造的 `run_end`。",
        "",
        "## 仍然有效的固定更新数结果",
        "",
        *rows,
        "",
        f"三方案共同拥有的最后检查点是第 {shared['step']} 步：纯 GS {shared['losses']['gradient_smoothing']:.6f}，常开 FE-E {shared['losses']['fe_e_always']:.6f}，GS + 门控 FE-E {shared['losses']['gs_fe_e_gated']:.6f}。",
        "",
        "纯 GS 首次达到 0.001 的检查点为第 "
        f"{gs['threshold_hits']['0.001']['step']} 步，损失 {gs['threshold_hits']['0.001']['loss']:.9f}。混合方案停止前尚未达到 0.001；常开 FE-E 到第 5000 步仍未发生有利学习相变。",
        "",
        "常开 FE-E 在第 3000→3125 步发生明显不利跃迁，验证损失从 3.0608 升至 3.4749，随后缓慢恢复并以 3.0239 结束。该结果表明持续传播平滑可能阻碍任务所需的层间功能重组。",
        "",
        "## 为什么墙钟比较作废",
        "",
        f"- 纯 GS 普通非探测步前 500 步中位数为 {gs['timing']['ordinary_first500_median']:.4f} 秒，末 500 步为 {gs['timing']['ordinary_last500_median']:.4f} 秒。",
        f"- 混合方案普通非介入步前 500 步中位数已升至 {hybrid['timing']['ordinary_first500_median']:.4f} 秒，末 500 步为 {hybrid['timing']['ordinary_last500_median']:.4f} 秒。",
        "- `pmset` 未记录正式 thermal/performance warning，但检查时 `mediaanalysisd`、Spotlight、Chrome Renderer 和 OrbStack 同时占用资源。细粒度 GPU/SoC 降频无法由本次只读检查确认。",
        "- 因此日志中的累计秒数仅保留为审计数据，不用于算法效率排序，也不与旧 AdamW 基线做墙钟配对。",
        "",
        "## 后续可接受的计时协议",
        "",
        "1. 重启并冷却机器，关闭或暂停非实验后台负载。",
        "2. 将三个方案按 125 或 250 步轮转交错运行，而不是整段串行，降低热态与时间顺序混杂。",
        "3. 每个方案先做独立预热，再报告 P50/P95 步时和达到固定损失阈值的时间。",
        "4. 至少重复三次轮转顺序；单次串行长跑不再作为效率证据。",
        "",
        "## 完整性",
        "",
        f"- 非有限值：GS {gs['nonfinite']}，常开 FE-E {methods['fe_e_always']['nonfinite']}，混合方案 {hybrid['nonfinite']}。",
        f"- 源码 SHA-256：`{summary['source_sha256']}`。",
        f"- 原始运行：`{summary['run']}`。",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    args = parser.parse_args()
    summary = audit(args.run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_report(summary), encoding="utf-8")
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
