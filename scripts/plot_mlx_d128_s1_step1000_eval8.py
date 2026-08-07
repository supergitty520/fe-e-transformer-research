#!/usr/bin/env python3
"""Plot the 128-layer, 1000-step, single-seed validation curves."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
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


def load_curves(path: Path) -> dict[str, list[tuple[int, float]]]:
    run = resolve_run(path)
    curves = {}
    for result_path in sorted((run / "runs").glob("*.json")):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result["variant"] in METHODS:
            curves[result["variant"]] = [
                (int(point["step"]), float(point["evaluation_loss"]))
                for point in result["evaluation_history"]
            ]
    missing = [method for method in METHODS if method not in curves]
    if missing:
        raise ValueError(f"missing curves: {missing}")
    return curves


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    curves = load_curves(args.run)

    width, height = 1440, 760
    left, right, top, bottom = 110, 1345, 78, 575
    x_min, x_max = 100.0, 1025.0
    y_min, y_max = 3.14, 3.49
    x_ticks = tuple(range(125, 1001, 125))
    y_ticks = (3.15, 3.20, 3.25, 3.30, 3.35, 3.40, 3.45)

    def xmap(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * (right - left)

    def ymap(value: float) -> float:
        return bottom - (value - y_min) / (y_max - y_min) * (bottom - top)

    shock_x = xmap(750)
    shock_y = ymap(dict(curves["fe_e_always"])[750])
    final_x = xmap(1000)
    final_y = ymap(dict(curves["gs_fe_e_gated"])[1000])

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#222}.title{font-size:24px;font-weight:600}.subtitle{font-size:14px;fill:#666}.tick{font-size:14px;fill:#555}.label{font-size:17px}.legend{font-size:14px;fill:#444}.note{font-size:14px;font-weight:600}</style>',
        f'<text class="title" x="{width / 2}" y="34" text-anchor="middle">128-layer Transformer — 1000 updates, seed 31</text>',
        f'<text class="subtitle" x="{width / 2}" y="57" text-anchor="middle">8 checkpoints × 8 validation batches; lower is better</text>',
    ]
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
        svg.append(f'<text class="tick" x="{x:.2f}" y="{bottom + 27}" text-anchor="middle">{tick}</text>')

    for method in METHODS:
        points = [(xmap(step), ymap(loss)) for step, loss in curves[method]]
        path = " ".join(("M" if index == 0 else "L") + f" {x:.2f} {y:.2f}" for index, (x, y) in enumerate(points))
        width_value = 4 if method == "gs_fe_e_gated" else 3
        svg.append(f'<path d="{path}" fill="none" stroke="{COLORS[method]}" stroke-width="{width_value}" stroke-linejoin="round"/>')
        for x, y in points:
            svg.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.5" fill="#fff" stroke="{COLORS[method]}" stroke-width="2.5"/>')

    svg.extend([
        f'<line x1="{shock_x:.2f}" x2="{shock_x - 58:.2f}" y1="{shock_y:.2f}" y2="{shock_y - 44:.2f}" stroke="#D55E00" stroke-width="1.5"/>',
        f'<text class="note" x="{shock_x - 64:.2f}" y="{shock_y - 51:.2f}" fill="#D55E00" text-anchor="end">FE-E transient +0.2032</text>',
        f'<line x1="{final_x:.2f}" x2="{final_x - 58:.2f}" y1="{final_y:.2f}" y2="{final_y - 42:.2f}" stroke="#CC79A7" stroke-width="1.5"/>',
        f'<text class="note" x="{final_x - 64:.2f}" y="{final_y - 48:.2f}" fill="#CC79A7" text-anchor="end">best final: 3.1585</text>',
        f'<text class="label" x="{(left + right) / 2}" y="{bottom + 65}" text-anchor="middle">Training update</text>',
        f'<text class="label" x="30" y="{(top + bottom) / 2}" text-anchor="middle" transform="rotate(-90 30 {(top + bottom) / 2})">Validation loss</text>',
    ])
    legend_y = 700
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
        "title": ImageFont.truetype(bold_path, 24 * scale),
        "subtitle": ImageFont.truetype(font_path, 14 * scale),
        "tick": ImageFont.truetype(font_path, 14 * scale),
        "label": ImageFont.truetype(font_path, 17 * scale),
        "legend": ImageFont.truetype(font_path, 14 * scale),
        "note": ImageFont.truetype(bold_path, 14 * scale),
    }

    def point(x: float, y: float) -> tuple[int, int]:
        return round(x * scale), round(y * scale)

    draw.text(point(width / 2, 29), "128-layer Transformer — 1000 updates, seed 31", font=fonts["title"], fill="#222", anchor="mm")
    draw.text(point(width / 2, 53), "8 checkpoints × 8 validation batches; lower is better", font=fonts["subtitle"], fill="#666", anchor="mm")
    for tick in y_ticks:
        y = ymap(tick)
        draw.line((*point(left, y), *point(right, y)), fill=(200, 200, 200, 150), width=scale)
        draw.text(point(left - 14, y), f"{tick:.2f}", font=fonts["tick"], fill="#555", anchor="rm")
    draw.line((*point(left, top), *point(left, bottom)), fill="#555", width=scale)
    draw.line((*point(left, bottom), *point(right, bottom)), fill="#555", width=scale)
    for tick in x_ticks:
        x = xmap(tick)
        draw.line((*point(x, bottom), *point(x, bottom + 6)), fill="#555", width=scale)
        draw.text(point(x, bottom + 21), str(tick), font=fonts["tick"], fill="#555", anchor="mm")
    for method in METHODS:
        points = [point(xmap(step), ymap(loss)) for step, loss in curves[method]]
        line_width = (4 if method == "gs_fe_e_gated" else 3) * scale
        draw.line(points, fill=COLORS[method], width=line_width, joint="curve")
        radius = round(4.5 * scale)
        for x, y in points:
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill="white", outline=COLORS[method], width=2 * scale)

    draw.line((*point(shock_x, shock_y), *point(shock_x - 58, shock_y - 44)), fill=COLORS["fe_e_always"], width=2 * scale)
    draw.text(point(shock_x - 64, shock_y - 51), "FE-E transient +0.2032", font=fonts["note"], fill=COLORS["fe_e_always"], anchor="rs")
    draw.line((*point(final_x, final_y), *point(final_x - 58, final_y - 42)), fill=COLORS["gs_fe_e_gated"], width=2 * scale)
    draw.text(point(final_x - 64, final_y - 48), "best final: 3.1585", font=fonts["note"], fill=COLORS["gs_fe_e_gated"], anchor="rs")
    draw.text(point((left + right) / 2, bottom + 56), "Training update", font=fonts["label"], fill="#333", anchor="mm")
    ylabel = "Validation loss"
    box = fonts["label"].getbbox(ylabel)
    layer = Image.new("RGBA", (box[2] - box[0] + 20, box[3] - box[1] + 20), (255, 255, 255, 0))
    ImageDraw.Draw(layer).text((10, 10), ylabel, font=fonts["label"], fill="#333")
    layer = layer.rotate(90, expand=True, resample=Image.Resampling.BICUBIC)
    image.paste(layer, point(23, (top + bottom) / 2 - layer.height / (2 * scale)), layer)
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
