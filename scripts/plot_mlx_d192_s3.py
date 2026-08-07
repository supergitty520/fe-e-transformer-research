#!/usr/bin/env python3
"""Plot mean ± SD curves for the formal 192-layer three-seed MLX run."""

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


def resolve_run(path: Path) -> Path:
    if (path / "manifest.json").exists():
        return path
    candidates = sorted(path.glob("run_*"))
    if not candidates:
        raise FileNotFoundError(f"no run_* directory under {path}")
    return candidates[-1]


def load_curves(path: Path) -> dict[str, list[tuple[int, float, float]]]:
    run = resolve_run(path)
    by_method: dict[str, list[dict]] = {method: [] for method in METHODS}
    for result_path in sorted((run / "runs").glob("*.json")):
        result = json.loads(result_path.read_text())
        if result["variant"] in by_method:
            by_method[result["variant"]].append(result)
    curves = {}
    for method in METHODS:
        results = sorted(by_method[method], key=lambda item: item["seed"])
        if len(results) != 3:
            raise ValueError(f"{method}: expected 3 seeds, found {len(results)}")
        points = []
        for index, checkpoint in enumerate(results[0]["evaluation_history"]):
            values = [float(result["evaluation_history"][index]["evaluation_loss"]) for result in results]
            points.append((int(checkpoint["step"]), statistics.fmean(values), statistics.stdev(values)))
        curves[method] = points
    return curves


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    curves = load_curves(args.run)

    width, height = 1400, 680
    left, right, top, bottom = 105, 1310, 74, 530
    x_min, x_max = 20.0, 255.0
    y_min, y_max = 3.32, 3.60
    x_ticks = tuple(range(25, 251, 25))
    y_ticks = (3.32, 3.36, 3.40, 3.44, 3.48, 3.52, 3.56, 3.60)

    def xmap(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * (right - left)

    def ymap(value: float) -> float:
        return bottom - (value - y_min) / (y_max - y_min) * (bottom - top)

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#222}.title{font-size:23px;font-weight:600}.tick{font-size:14px;fill:#555}.label{font-size:17px}.legend{font-size:14px;fill:#444}.note{font-size:13px;fill:#666}</style>',
        f'<text class="title" x="{width / 2}" y="33" text-anchor="middle">192-layer Transformer — 250 steps, 3 seeds, 5 validation batches</text>',
    ]
    tail_x = xmap(200)
    svg.append(f'<rect x="{tail_x:.2f}" y="{top}" width="{right - tail_x:.2f}" height="{bottom - top}" fill="#808080" fill-opacity="0.055"/>')
    svg.append(f'<text class="note" x="{(tail_x + right) / 2:.2f}" y="{top + 18}" text-anchor="middle">tail-3 sensitivity window</text>')
    for tick in y_ticks:
        y = ymap(tick)
        svg.append(f'<line x1="{left}" x2="{right}" y1="{y:.2f}" y2="{y:.2f}" stroke="#c8c8c8" stroke-opacity="0.55"/>')
        svg.append(f'<text class="tick" x="{left - 14}" y="{y + 5:.2f}" text-anchor="end">{tick:.2f}</text>')
    svg.extend([
        f'<line x1="{left}" x2="{left}" y1="{top}" y2="{bottom}" stroke="#555"/>',
        f'<line x1="{left}" x2="{right}" y1="{bottom}" y2="{bottom}" stroke="#555"/>',
    ])
    for tick in x_ticks:
        x = xmap(tick)
        svg.append(f'<line x1="{x:.2f}" x2="{x:.2f}" y1="{bottom}" y2="{bottom + 6}" stroke="#555"/>')
        svg.append(f'<text class="tick" x="{x:.2f}" y="{bottom + 26}" text-anchor="middle">{tick}</text>')

    for method in METHODS:
        values = curves[method]
        upper = [(xmap(step), ymap(mean + sd)) for step, mean, sd in values]
        lower = [(xmap(step), ymap(mean - sd)) for step, mean, sd in reversed(values)]
        polygon = " ".join(f"{x:.2f},{y:.2f}" for x, y in upper + lower)
        points = [(xmap(step), ymap(mean)) for step, mean, _ in values]
        path = " ".join(("M" if index == 0 else "L") + f" {x:.2f} {y:.2f}" for index, (x, y) in enumerate(points))
        svg.append(f'<polygon points="{polygon}" fill="{COLORS[method]}" fill-opacity="0.10"/>')
        svg.append(f'<path d="{path}" fill="none" stroke="{COLORS[method]}" stroke-width="3" stroke-linejoin="round"/>')
        for x, y in points:
            svg.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="#fff" stroke="{COLORS[method]}" stroke-width="2.4"/>')
    svg.extend([
        f'<text class="label" x="{(left + right) / 2}" y="{bottom + 62}" text-anchor="middle">Training step</text>',
        f'<text class="label" x="28" y="{(top + bottom) / 2}" text-anchor="middle" transform="rotate(-90 28 {(top + bottom) / 2})">Validation loss (mean ± SD, n=3)</text>',
    ])
    legend_y = 646
    widths = (145, 235, 185, 230, 205)
    cursor = (width - sum(widths)) / 2
    for method, item_width in zip(METHODS, widths):
        svg.append(f'<line x1="{cursor:.2f}" x2="{cursor + 28:.2f}" y1="{legend_y}" y2="{legend_y}" stroke="{COLORS[method]}" stroke-width="3"/>')
        svg.append(f'<circle cx="{cursor + 14:.2f}" cy="{legend_y}" r="4" fill="#fff" stroke="{COLORS[method]}" stroke-width="2.4"/>')
        svg.append(f'<text class="legend" x="{cursor + 37:.2f}" y="{legend_y + 5}">{escape(LABELS[method])}</text>')
        cursor += item_width
    svg.append("</svg>")

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    args.output_prefix.with_suffix(".svg").write_text("\n".join(svg) + "\n", encoding="utf-8")

    scale = 2
    image = Image.new("RGB", (width * scale, height * scale), "white")
    draw = ImageDraw.Draw(image, "RGBA")
    font_path = "/System/Library/Fonts/Supplemental/Arial.ttf"
    bold_path = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
    fonts = {
        "title": ImageFont.truetype(bold_path, 23 * scale),
        "tick": ImageFont.truetype(font_path, 14 * scale),
        "label": ImageFont.truetype(font_path, 17 * scale),
        "legend": ImageFont.truetype(font_path, 14 * scale),
        "note": ImageFont.truetype(font_path, 13 * scale),
    }

    def point(x: float, y: float) -> tuple[int, int]:
        return round(x * scale), round(y * scale)

    draw.text(point(width / 2, 27), "192-layer Transformer — 250 steps, 3 seeds, 5 validation batches", font=fonts["title"], fill="#222", anchor="mm")
    draw.rectangle((*point(tail_x, top), *point(right, bottom)), fill=(128, 128, 128, 14))
    draw.text(point((tail_x + right) / 2, top + 13), "tail-3 sensitivity window", font=fonts["note"], fill="#666", anchor="mm")
    for tick in y_ticks:
        y = ymap(tick)
        draw.line((*point(left, y), *point(right, y)), fill=(200, 200, 200, 150), width=scale)
        draw.text(point(left - 14, y), f"{tick:.2f}", font=fonts["tick"], fill="#555", anchor="rm")
    draw.line((*point(left, top), *point(left, bottom)), fill="#555", width=scale)
    draw.line((*point(left, bottom), *point(right, bottom)), fill="#555", width=scale)
    for tick in x_ticks:
        x = xmap(tick)
        draw.line((*point(x, bottom), *point(x, bottom + 6)), fill="#555", width=scale)
        draw.text(point(x, bottom + 20), str(tick), font=fonts["tick"], fill="#555", anchor="mm")
    for method in METHODS:
        values = curves[method]
        upper = [point(xmap(step), ymap(mean + sd)) for step, mean, sd in values]
        lower = [point(xmap(step), ymap(mean - sd)) for step, mean, sd in reversed(values)]
        color = COLORS[method]
        red, green, blue = tuple(int(color[index:index + 2], 16) for index in (1, 3, 5))
        draw.polygon(upper + lower, fill=(red, green, blue, 26))
        points = [point(xmap(step), ymap(mean)) for step, mean, _ in values]
        draw.line(points, fill=color, width=3 * scale, joint="curve")
        radius = 4 * scale
        for x, y in points:
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill="white", outline=color, width=2 * scale)
    draw.text(point((left + right) / 2, bottom + 54), "Training step", font=fonts["label"], fill="#333", anchor="mm")
    ylabel = "Validation loss (mean ± SD, n=3)"
    box = fonts["label"].getbbox(ylabel)
    label_layer = Image.new("RGBA", (box[2] - box[0] + 20, box[3] - box[1] + 20), (255, 255, 255, 0))
    ImageDraw.Draw(label_layer).text((10, 10), ylabel, font=fonts["label"], fill="#333")
    label_layer = label_layer.rotate(90, expand=True, resample=Image.Resampling.BICUBIC)
    image.paste(label_layer, point(22, (top + bottom) / 2 - label_layer.height / (2 * scale)), label_layer)
    cursor = (width - sum(widths)) / 2
    for method, item_width in zip(METHODS, widths):
        draw.line((*point(cursor, legend_y), *point(cursor + 28, legend_y)), fill=COLORS[method], width=3 * scale)
        x, y = point(cursor + 14, legend_y)
        radius = 4 * scale
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill="white", outline=COLORS[method], width=2 * scale)
        draw.text(point(cursor + 37, legend_y), LABELS[method], font=fonts["legend"], fill="#444", anchor="lm")
        cursor += item_width
    image.save(args.output_prefix.with_suffix(".png"), dpi=(240, 240))


if __name__ == "__main__":
    main()
