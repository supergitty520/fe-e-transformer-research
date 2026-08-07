#!/usr/bin/env python3
"""Plot three-seed MLX D96 validation-loss curves for paper use."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
from xml.sax.saxutils import escape

from PIL import Image, ImageDraw, ImageFont


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
    "fe_e_always": "FE-E always-on",
    "fe_e_gated": "Observer-gated FE-E",
    "gs_fe_e_gated": "GS + gated FE-E",
}
COLORS = {
    "baseline": "#595959",
    "gradient_smoothing": "#0072B2",
    "fe_e_always": "#D55E00",
    "fe_e_gated": "#009E73",
    "gs_fe_e_gated": "#CC79A7",
}
MARKERS = {
    "baseline": "o",
    "gradient_smoothing": "s",
    "fe_e_always": "^",
    "fe_e_gated": "D",
    "gs_fe_e_gated": "v",
}


def resolve_run(path: Path) -> Path:
    if (path / "manifest.json").exists():
        return path
    candidates = sorted(path.glob("run_*"))
    if not candidates:
        raise FileNotFoundError(f"no run_* directory under {path}")
    return candidates[-1]


def load_curves(path: Path) -> dict[str, tuple[list[int], list[float], list[float]]]:
    run = resolve_run(path)
    by_method: dict[str, list[dict]] = {method: [] for method in METHODS}
    for result_path in sorted((run / "runs").glob("*.json")):
        result = json.loads(result_path.read_text())
        if result["variant"] in by_method:
            by_method[result["variant"]].append(result)

    curves = {}
    for method in METHODS:
        runs = sorted(by_method[method], key=lambda item: item["seed"])
        if len(runs) != 3:
            raise ValueError(f"{method}: expected 3 seeds, found {len(runs)}")
        steps = [int(item["step"]) for item in runs[0]["evaluation_history"]]
        means = []
        deviations = []
        for index in range(len(steps)):
            values = [float(run["evaluation_history"][index]["evaluation_loss"]) for run in runs]
            means.append(statistics.fmean(values))
            deviations.append(statistics.stdev(values))
        curves[method] = (steps, means, deviations)
    return curves


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", type=Path, required=True)
    parser.add_argument("--stress", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    scenarios = (
        ("Clean training", load_curves(args.clean)),
        ("Learning-rate shock", load_curves(args.stress)),
    )
    width, height = 1400, 620
    plot_top, plot_bottom = 76, 500
    panel_width, gap = 580, 92
    panel_lefts = (96, 96 + panel_width + gap)
    x_min, x_max = 20.0, 205.0
    y_min, y_max = 3.30, 3.60

    def xmap(value: float, left: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * panel_width

    def ymap(value: float) -> float:
        return plot_bottom - (value - y_min) / (y_max - y_min) * (plot_bottom - plot_top)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#222} .tick{font-size:13px;fill:#555} .label{font-size:15px;fill:#333} .title{font-size:19px;font-weight:600} .panel{font-size:17px;font-weight:600}</style>',
        f'<text class="title" x="{width / 2}" y="30" text-anchor="middle">96-layer Transformer propagation constraints</text>',
    ]
    y_ticks = (3.30, 3.35, 3.40, 3.45, 3.50, 3.55, 3.60)
    x_ticks = (25, 50, 75, 100, 125, 150, 175, 200)

    for panel_index, ((title, curves), left) in enumerate(zip(scenarios, panel_lefts)):
        lines.append(f'<text class="panel" x="{left + panel_width / 2}" y="59" text-anchor="middle">{escape(title)}</text>')
        if panel_index == 1:
            shock_x = xmap(80, left)
            shock_width = xmap(88, left) - shock_x
            lines.append(
                f'<rect x="{shock_x:.2f}" y="{plot_top}" width="{shock_width:.2f}" height="{plot_bottom - plot_top}" fill="#E69F00" fill-opacity="0.16"/>'
            )
            lines.append(f'<text class="tick" x="{shock_x + shock_width / 2:.2f}" y="92" text-anchor="middle" fill="#8A5600">5× LR</text>')
        for tick in y_ticks:
            y = ymap(tick)
            lines.append(f'<line x1="{left}" x2="{left + panel_width}" y1="{y:.2f}" y2="{y:.2f}" stroke="#c8c8c8" stroke-opacity="0.55"/>')
            if panel_index == 0:
                lines.append(f'<text class="tick" x="{left - 12}" y="{y + 5:.2f}" text-anchor="end">{tick:.2f}</text>')
        lines.append(f'<line x1="{left}" x2="{left}" y1="{plot_top}" y2="{plot_bottom}" stroke="#555"/>')
        lines.append(f'<line x1="{left}" x2="{left + panel_width}" y1="{plot_bottom}" y2="{plot_bottom}" stroke="#555"/>')
        for tick in x_ticks:
            x = xmap(tick, left)
            lines.append(f'<line x1="{x:.2f}" x2="{x:.2f}" y1="{plot_bottom}" y2="{plot_bottom + 5}" stroke="#555"/>')
            lines.append(f'<text class="tick" x="{x:.2f}" y="{plot_bottom + 23}" text-anchor="middle">{tick}</text>')

        for method in METHODS:
            steps, means, deviations = curves[method]
            upper = [(xmap(step, left), ymap(mean + deviation)) for step, mean, deviation in zip(steps, means, deviations)]
            lower = [(xmap(step, left), ymap(mean - deviation)) for step, mean, deviation in zip(reversed(steps), reversed(means), reversed(deviations))]
            band = " ".join(f"{x:.2f},{y:.2f}" for x, y in upper + lower)
            mean_points = [(xmap(step, left), ymap(mean)) for step, mean in zip(steps, means)]
            path = " ".join(("M" if index == 0 else "L") + f" {x:.2f} {y:.2f}" for index, (x, y) in enumerate(mean_points))
            lines.append(f'<polygon points="{band}" fill="{COLORS[method]}" fill-opacity="0.10"/>')
            lines.append(f'<path d="{path}" fill="none" stroke="{COLORS[method]}" stroke-width="2.4" stroke-linejoin="round"/>')
            for x, y in mean_points:
                lines.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.4" fill="#fff" stroke="{COLORS[method]}" stroke-width="2"/>')

        lines.append(f'<text class="label" x="{left + panel_width / 2}" y="{plot_bottom + 49}" text-anchor="middle">Training step</text>')

    lines.append(f'<text class="label" x="24" y="{(plot_top + plot_bottom) / 2}" text-anchor="middle" transform="rotate(-90 24 {(plot_top + plot_bottom) / 2})">Validation loss (mean ± SD, n=3)</text>')
    legend_y = 584
    item_widths = (130, 230, 180, 230, 200)
    total = sum(item_widths)
    cursor = (width - total) / 2
    for method, item_width in zip(METHODS, item_widths):
        lines.append(f'<line x1="{cursor:.2f}" x2="{cursor + 25:.2f}" y1="{legend_y}" y2="{legend_y}" stroke="{COLORS[method]}" stroke-width="3"/>')
        lines.append(f'<circle cx="{cursor + 12.5:.2f}" cy="{legend_y}" r="3.4" fill="#fff" stroke="{COLORS[method]}" stroke-width="2"/>')
        lines.append(f'<text class="tick" x="{cursor + 33:.2f}" y="{legend_y + 5}">{escape(LABELS[method])}</text>')
        cursor += item_width
    lines.append("</svg>")

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    args.output_prefix.with_suffix(".svg").write_text("\n".join(lines) + "\n", encoding="utf-8")

    scale = 2
    image = Image.new("RGB", (width * scale, height * scale), "white")
    draw = ImageDraw.Draw(image, "RGBA")
    font_path = "/System/Library/Fonts/Supplemental/Arial.ttf"
    bold_path = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
    font_tick = ImageFont.truetype(font_path, 13 * scale)
    font_label = ImageFont.truetype(font_path, 15 * scale)
    font_panel = ImageFont.truetype(bold_path, 17 * scale)
    font_title = ImageFont.truetype(bold_path, 19 * scale)

    def point(x: float, y: float) -> tuple[int, int]:
        return round(x * scale), round(y * scale)

    draw.text(point(width / 2, 25), "96-layer Transformer propagation constraints", font=font_title, fill="#222", anchor="mm")
    for panel_index, ((title, curves), left) in enumerate(zip(scenarios, panel_lefts)):
        draw.text(point(left + panel_width / 2, 54), title, font=font_panel, fill="#222", anchor="mm")
        if panel_index == 1:
            shock_x = xmap(80, left)
            shock_right = xmap(88, left)
            draw.rectangle((*point(shock_x, plot_top), *point(shock_right, plot_bottom)), fill=(230, 159, 0, 40))
            draw.text(point((shock_x + shock_right) / 2, 87), "5× LR", font=font_tick, fill="#8A5600", anchor="mm")
        for tick in y_ticks:
            y = ymap(tick)
            draw.line((*point(left, y), *point(left + panel_width, y)), fill=(200, 200, 200, 140), width=scale)
            if panel_index == 0:
                draw.text(point(left - 12, y), f"{tick:.2f}", font=font_tick, fill="#555", anchor="rm")
        draw.line((*point(left, plot_top), *point(left, plot_bottom)), fill="#555", width=scale)
        draw.line((*point(left, plot_bottom), *point(left + panel_width, plot_bottom)), fill="#555", width=scale)
        for tick in x_ticks:
            x = xmap(tick, left)
            draw.line((*point(x, plot_bottom), *point(x, plot_bottom + 5)), fill="#555", width=scale)
            draw.text(point(x, plot_bottom + 17), str(tick), font=font_tick, fill="#555", anchor="mm")
        for method in METHODS:
            steps, means, deviations = curves[method]
            upper = [point(xmap(step, left), ymap(mean + deviation)) for step, mean, deviation in zip(steps, means, deviations)]
            lower = [point(xmap(step, left), ymap(mean - deviation)) for step, mean, deviation in zip(reversed(steps), reversed(means), reversed(deviations))]
            color = COLORS[method]
            red, green, blue = tuple(int(color[index:index + 2], 16) for index in (1, 3, 5))
            draw.polygon(upper + lower, fill=(red, green, blue, 26))
            mean_points = [point(xmap(step, left), ymap(mean)) for step, mean in zip(steps, means)]
            draw.line(mean_points, fill=color, width=round(2.4 * scale), joint="curve")
            radius = round(3.4 * scale)
            for x, y in mean_points:
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill="white", outline=color, width=2 * scale)
        draw.text(point(left + panel_width / 2, plot_bottom + 43), "Training step", font=font_label, fill="#333", anchor="mm")

    y_label = "Validation loss (mean ± SD, n=3)"
    bbox = font_label.getbbox(y_label)
    label_image = Image.new("RGBA", (bbox[2] - bbox[0] + 12 * scale, bbox[3] - bbox[1] + 12 * scale), (255, 255, 255, 0))
    ImageDraw.Draw(label_image).text((6 * scale, 6 * scale), y_label, font=font_label, fill="#333")
    label_image = label_image.rotate(90, expand=True, resample=Image.Resampling.BICUBIC)
    image.paste(label_image, point(23, (plot_top + plot_bottom) / 2 - label_image.height / (2 * scale)), label_image)

    legend_y = 584
    cursor = (width - total) / 2
    for method, item_width in zip(METHODS, item_widths):
        draw.line((*point(cursor, legend_y), *point(cursor + 25, legend_y)), fill=COLORS[method], width=3 * scale)
        x, y = point(cursor + 12.5, legend_y)
        radius = round(3.4 * scale)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill="white", outline=COLORS[method], width=2 * scale)
        draw.text(point(cursor + 33, legend_y), LABELS[method], font=font_tick, fill="#555", anchor="lm")
        cursor += item_width
    image.save(args.output_prefix.with_suffix(".png"), dpi=(240, 240))


if __name__ == "__main__":
    main()
