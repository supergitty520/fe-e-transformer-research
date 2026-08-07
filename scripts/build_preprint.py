#!/usr/bin/env python3
"""Build the FE-E preprint PDF from its Markdown source.

This intentionally uses ReportLab rather than a hidden online renderer so the
paper can be rebuilt offline inside the bundled Codex PDF runtime.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import html
import json
import math
from pathlib import Path
import re
import statistics

from reportlab.graphics.shapes import Circle, Drawing, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Flowable,
    Frame,
    KeepTogether,
    PageTemplate,
    PageBreak,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper" / "fe_e_preprint.md"
RESULTS = ROOT / "results"

NAVY = HexColor("#102A43")
BLUE = HexColor("#2F6BFF")
CYAN = HexColor("#18A5A7")
ORANGE = HexColor("#E57A44")
INK = HexColor("#17212B")
MUTED = HexColor("#52606D")
GRID = HexColor("#CBD5E1")
PALE = HexColor("#F2F6FA")
PALE_BLUE = HexColor("#EAF0FF")
PALE_CYAN = HexColor("#E8F7F6")
PALE_ORANGE = HexColor("#FCEFE8")


def register_fonts() -> None:
    folder = Path("/System/Library/Fonts/Supplemental")
    pdfmetrics.registerFont(TTFont("TNR", str(folder / "Times New Roman.ttf")))
    pdfmetrics.registerFont(TTFont("TNR-Bold", str(folder / "Times New Roman Bold.ttf")))
    pdfmetrics.registerFont(TTFont("TNR-Italic", str(folder / "Times New Roman Italic.ttf")))
    pdfmetrics.registerFont(
        TTFont("TNR-BoldItalic", str(folder / "Times New Roman Bold Italic.ttf"))
    )
    pdfmetrics.registerFont(TTFont("Arial", str(folder / "Arial.ttf")))
    pdfmetrics.registerFont(TTFont("Arial-Bold", str(folder / "Arial Bold.ttf")))
    pdfmetrics.registerFont(TTFont("Arial-Italic", str(folder / "Arial Italic.ttf")))
    pdfmetrics.registerFontFamily(
        "TNR", normal="TNR", bold="TNR-Bold", italic="TNR-Italic", boldItalic="TNR-BoldItalic"
    )


def styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "Title", parent=base["Title"], fontName="Arial-Bold", fontSize=22,
            leading=25, textColor=NAVY, alignment=TA_LEFT, spaceAfter=10,
        ),
        "author": ParagraphStyle(
            "Author", fontName="Arial-Bold", fontSize=11.5, leading=15,
            textColor=INK, spaceAfter=1,
        ),
        "meta": ParagraphStyle(
            "Meta", fontName="Arial", fontSize=8.8, leading=12, textColor=MUTED,
        ),
        "h1": ParagraphStyle(
            "H1", fontName="Arial-Bold", fontSize=15, leading=18, textColor=NAVY,
            spaceBefore=15, spaceAfter=6, keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "H2", fontName="Arial-Bold", fontSize=11.3, leading=14, textColor=BLUE,
            spaceBefore=10, spaceAfter=4, keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "Body", fontName="TNR", fontSize=9.6, leading=12.8, textColor=INK,
            alignment=TA_JUSTIFY, spaceAfter=6, splitLongWords=False,
        ),
        "abstract": ParagraphStyle(
            "Abstract", fontName="TNR", fontSize=9.25, leading=12.4,
            textColor=INK, alignment=TA_JUSTIFY, backColor=PALE,
            borderColor=GRID, borderWidth=0.6, borderPadding=10, spaceAfter=12,
        ),
        "bullet": ParagraphStyle(
            "Bullet", fontName="TNR", fontSize=9.4, leading=12.4, textColor=INK,
            leftIndent=17, firstLineIndent=-9, bulletIndent=5, spaceAfter=3,
        ),
        "equation": ParagraphStyle(
            "Equation", fontName="TNR-Italic", fontSize=10.3, leading=14,
            textColor=INK, alignment=TA_CENTER, leftIndent=12, rightIndent=12,
        ),
        "caption": ParagraphStyle(
            "Caption", fontName="Arial", fontSize=7.8, leading=10.2,
            textColor=MUTED, alignment=TA_LEFT, spaceBefore=3, spaceAfter=8,
        ),
        "table_caption": ParagraphStyle(
            "TableCaption", fontName="Arial-Bold", fontSize=8.1, leading=10.2,
            textColor=NAVY, spaceBefore=5, spaceAfter=4, keepWithNext=True,
        ),
        "table_header": ParagraphStyle(
            "TableHeader", fontName="Arial-Bold", fontSize=7.3, leading=9.2,
            textColor=colors.white, alignment=TA_CENTER,
        ),
        "code": ParagraphStyle(
            "Code", fontName="Courier", fontSize=7.7, leading=10, textColor=INK,
            backColor=PALE, borderColor=GRID, borderWidth=0.5, borderPadding=7,
            leftIndent=4, rightIndent=4, spaceBefore=3, spaceAfter=8,
        ),
        "reference": ParagraphStyle(
            "Reference", fontName="TNR", fontSize=7.6, leading=9.25,
            textColor=INK, alignment=TA_LEFT, leftIndent=13, firstLineIndent=-13,
            spaceAfter=2.5,
        ),
    }


def inline(text: str) -> str:
    text = text.replace("+/-", "±")
    escaped = html.escape(text, quote=False)
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)
    escaped = re.sub(r"`([^`]+)`", r'<font name="Courier">\1</font>', escaped)
    escaped = re.sub(
        r"(https?://[^\s<]+)",
        lambda match: f'<link href="{match.group(1)}" color="#2F6BFF">{match.group(1)}</link>',
        escaped,
    )
    return escaped


def equation_html(text: str) -> str:
    replacements = {
        "partial": "∂", "Delta": "Δ", "sum": "Σ", "tensor": "⊗",
        "epsilon": "ε", "lambda": "λ", "alpha": "α", "theta": "θ",
        "infinity": "∞", "<=": "≤",
    }
    result = " ".join(text.strip().split())
    for old, new in replacements.items():
        result = result.replace(old, new)
    result = html.escape(result, quote=False)
    result = re.sub(r"([A-Za-zΕΔΣλθα∂]+)_\{?([A-Za-z0-9,+-]+)\}?", r"\1<sub>\2</sub>", result)
    result = re.sub(r"\^\(([^)]+)\)", r"<super>\1</super>", result)
    result = re.sub(r"\^\{?([A-Za-z0-9,+-]+)\}?", r"<super>\1</super>", result)
    return result


class Rule(Flowable):
    def __init__(self, width: float, color=BLUE, thickness: float = 1.5):
        super().__init__()
        self.width = width
        self.height = thickness + 4
        self.color = color
        self.thickness = thickness

    def draw(self) -> None:
        self.canv.setStrokeColor(self.color)
        self.canv.setLineWidth(self.thickness)
        self.canv.line(0, self.height - 2, self.width, self.height - 2)


def draw_overview(width: float) -> Drawing:
    height = 160
    d = Drawing(width, height)
    d.add(Rect(0, 0, width, height, fillColor=colors.white, strokeColor=GRID, rx=6, ry=6))
    d.add(String(14, 140, "Hidden-state adjoints as a finite-element field over depth",
                 fontName="Arial-Bold", fontSize=10, fillColor=NAVY))
    x0, x1, y = 32, width - 32, 111
    count = 8
    gap = (x1 - x0) / (count - 1)
    d.add(Line(x0, y, x1, y, strokeColor=MUTED, strokeWidth=1.2))
    for index in range(count):
        x = x0 + index * gap
        d.add(Circle(x, y, 7, fillColor=colors.white, strokeColor=BLUE, strokeWidth=1.5))
        d.add(String(x - 6, y + 12, f"g{index}", fontName="TNR-Italic", fontSize=8, fillColor=INK))
    boxes = [
        (14, 20, (width - 40) / 3, 58, PALE_BLUE, BLUE, "STIFFNESS", "local jumps and rotations"),
        (20 + (width - 40) / 3, 20, (width - 40) / 3, 58, PALE_CYAN, CYAN, "MASS", "integrated amplitude anchor"),
        (26 + 2 * (width - 40) / 3, 20, (width - 40) / 3, 58, PALE_ORANGE, ORANGE, "ENTROPY", "depth-wise concentration"),
    ]
    for x, by, bw, bh, fill, stroke, title, subtitle in boxes:
        d.add(Rect(x, by, bw - 6, bh, fillColor=fill, strokeColor=stroke, rx=4, ry=4))
        d.add(String(x + 9, by + 36, title, fontName="Arial-Bold", fontSize=8.2, fillColor=stroke))
        d.add(String(x + 9, by + 20, subtitle, fontName="Arial", fontSize=7.2, fillColor=INK))
        d.add(Line(x + (bw - 6) / 2, by + bh, x + (bw - 6) / 2, y - 10,
                   strokeColor=stroke, strokeWidth=0.8))
    return d


def load_confirmation() -> dict[str, dict[int, tuple[float, float]]]:
    seeds = [31, 47, 59, 71, 89]
    collected: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for seed in seeds:
        payload = json.loads((RESULTS / f"direct_compare_d24_seed{seed}.json").read_text())
        for run in payload["results"]:
            for point in run["evaluation_history"]:
                collected[run["variant"]][int(point["step"])].append(float(point["evaluation_loss"]))
    return {
        variant: {
            step: (statistics.fmean(values), statistics.stdev(values))
            for step, values in step_values.items()
        }
        for variant, step_values in collected.items()
    }


def draw_loss_curves(width: float) -> Drawing:
    data = load_confirmation()
    height = 230
    d = Drawing(width, height)
    d.add(Rect(0, 0, width, height, fillColor=colors.white, strokeColor=GRID, rx=5, ry=5))
    left, right, bottom, top = 48, width - 18, 38, height - 24
    y_min, y_max = 2.96, 3.52
    x_min, x_max = 25, 200

    def xp(step: float) -> float:
        return left + (step - x_min) / (x_max - x_min) * (right - left)

    def yp(value: float) -> float:
        return bottom + (value - y_min) / (y_max - y_min) * (top - bottom)

    for value in [3.0, 3.1, 3.2, 3.3, 3.4, 3.5]:
        yy = yp(value)
        d.add(Line(left, yy, right, yy, strokeColor=HexColor("#E5EAF0"), strokeWidth=0.6))
        d.add(String(left - 31, yy - 3, f"{value:.1f}", fontName="Arial", fontSize=7, fillColor=MUTED))
    for step in [25, 50, 75, 100, 125, 150, 175, 200]:
        xx = xp(step)
        d.add(Line(xx, bottom, xx, bottom - 3, strokeColor=MUTED, strokeWidth=0.6))
        d.add(String(xx - 7, bottom - 14, str(step), fontName="Arial", fontSize=6.8, fillColor=MUTED))
    d.add(Line(left, bottom, right, bottom, strokeColor=INK, strokeWidth=0.9))
    d.add(Line(left, bottom, left, top, strokeColor=INK, strokeWidth=0.9))
    d.add(String((left + right) / 2 - 18, 12, "Update step", fontName="Arial", fontSize=7.5, fillColor=MUTED))
    d.add(String(8, top - 3, "Eval. loss", fontName="Arial", fontSize=7.5, fillColor=MUTED))

    specs = [
        ("baseline", "AdamW", MUTED, HexColor("#E3E8ED")),
        ("gradient_smoothing", "Gradient Smoothing", BLUE, PALE_BLUE),
        ("fe_entropy", "FE-E", ORANGE, PALE_ORANGE),
    ]
    for variant, label, color, fill in specs:
        points = sorted(data[variant])
        upper = [(xp(step), yp(data[variant][step][0] + data[variant][step][1])) for step in points]
        lower = [(xp(step), yp(data[variant][step][0] - data[variant][step][1])) for step in reversed(points)]
        polygon = [coordinate for point in upper + lower for coordinate in point]
        d.add(Polygon(polygon, fillColor=fill, strokeColor=None))
        previous = None
        for step in points:
            point = (xp(step), yp(data[variant][step][0]))
            if previous is not None:
                d.add(Line(previous[0], previous[1], point[0], point[1], strokeColor=color, strokeWidth=1.8))
            d.add(Circle(point[0], point[1], 2.2, fillColor=colors.white, strokeColor=color, strokeWidth=1.2))
            previous = point
    legend_y = top - 8
    legend_x = left + 8
    for _, label, color, _ in specs:
        d.add(Line(legend_x, legend_y, legend_x + 16, legend_y, strokeColor=color, strokeWidth=2))
        d.add(String(legend_x + 21, legend_y - 3, label, fontName="Arial", fontSize=7.2, fillColor=INK))
        legend_x += 110 if label != "Gradient Smoothing" else 135
    return d


def draw_tradeoff(width: float) -> Drawing:
    height = 220
    d = Drawing(width, height)
    d.add(Rect(0, 0, width, height, fillColor=colors.white, strokeColor=GRID, rx=5, ry=5))
    methods = ["AdamW", "Grad. Smooth.", "FE-E"]
    method_colors = [MUTED, BLUE, ORANGE]
    fixed = [3.1121, 3.0911, 3.0103]
    compute = [3.1121, 3.0911, 3.1770]
    costs = [1.0, 1.26, 2.29]

    panel_w = (width - 34) / 2
    left_x = 16
    base_y, top_y = 38, height - 34
    d.add(String(left_x, height - 19, "Evaluation loss", fontName="Arial-Bold", fontSize=9, fillColor=NAVY))
    y_min, y_max = 2.98, 3.20
    for val in [3.0, 3.1, 3.2]:
        yy = base_y + (val - y_min) / (y_max - y_min) * (top_y - base_y)
        d.add(Line(left_x + 28, yy, left_x + panel_w - 6, yy, strokeColor=HexColor("#E5EAF0"), strokeWidth=0.5))
        d.add(String(left_x, yy - 3, f"{val:.1f}", fontName="Arial", fontSize=6.8, fillColor=MUTED))
    group_w = (panel_w - 48) / 3
    for i, method in enumerate(methods):
        gx = left_x + 34 + i * group_w
        for j, values in enumerate((fixed, compute)):
            val = values[i]
            yy = base_y + (val - y_min) / (y_max - y_min) * (top_y - base_y)
            fill = method_colors[i] if j == 0 else colors.white
            d.add(Rect(gx + j * 12, base_y, 9, yy - base_y, fillColor=fill,
                       strokeColor=method_colors[i], strokeWidth=0.9))
        d.add(String(gx - 3, 22, method, fontName="Arial", fontSize=6.7, fillColor=INK))
    d.add(Rect(left_x + 44, top_y + 6, 8, 8, fillColor=NAVY, strokeColor=NAVY))
    d.add(String(left_x + 56, top_y + 7, "fixed steps", fontName="Arial", fontSize=6.8, fillColor=INK))
    d.add(Rect(left_x + 117, top_y + 6, 8, 8, fillColor=colors.white, strokeColor=NAVY))
    d.add(String(left_x + 129, top_y + 7, "compute proxy", fontName="Arial", fontSize=6.8, fillColor=INK))

    right_x = left_x + panel_w + 18
    d.add(String(right_x, height - 19, "Observed step-time ratio", fontName="Arial-Bold", fontSize=9, fillColor=NAVY))
    r_left, r_right = right_x + 25, width - 16
    for val in [0, 1, 2, 3]:
        yy = base_y + val / 3 * (top_y - base_y)
        d.add(Line(r_left, yy, r_right, yy, strokeColor=HexColor("#E5EAF0"), strokeWidth=0.5))
        d.add(String(right_x + 8, yy - 3, str(val), fontName="Arial", fontSize=6.8, fillColor=MUTED))
    rw = (r_right - r_left) / 3
    for i, (method, value, color) in enumerate(zip(methods, costs, method_colors, strict=True)):
        bx = r_left + i * rw + rw * 0.25
        yy = base_y + value / 3 * (top_y - base_y)
        d.add(Rect(bx, base_y, rw * 0.48, yy - base_y, fillColor=color, strokeColor=color))
        d.add(String(bx + 3, yy + 4, f"{value:.2f}x", fontName="Arial-Bold", fontSize=7, fillColor=color))
        d.add(String(bx - 2, 22, method, fontName="Arial", fontSize=6.7, fillColor=INK))
    return d


FIGURES = {
    "overview": (draw_overview, "Figure 1. FE-E treats task-loss gradients with respect to residual states as a depth-wise finite-element field. Stiffness controls local shape, mass controls absolute energy, and relative entropy controls concentration."),
    "loss_curves": (draw_loss_curves, "Figure 2. Mean evaluation loss over five confirmation seeds; shaded regions show one sample standard deviation. FE-E has the best fixed-step curve in this short synthetic experiment."),
    "tradeoff": (draw_tradeoff, "Figure 3. Fixed-step quality and compute efficiency separate. Filled loss bars use equal update counts; outlined bars use the approximate backward-count budget. Exact FE-E has the strongest per-update result but the highest observed cost."),
}


def table_from_lines(lines: list[str], usable_width: float, style_map: dict[str, ParagraphStyle]) -> Table:
    rows: list[list[str]] = []
    for line in lines:
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if all(re.fullmatch(r":?-+:?", cell) for cell in cells):
            continue
        rows.append(cells)
    paragraphs = [
        [Paragraph(inline(cell), style_map["table_header"] if row_index == 0 else style_map["body"])
         for cell in row]
        for row_index, row in enumerate(rows)
    ]
    columns = len(paragraphs[0])
    first_fraction = 0.20 if columns >= 5 else 0.24
    remaining = (1.0 - first_fraction) / (columns - 1)
    widths = [usable_width * first_fraction] + [usable_width * remaining] * (columns - 1)
    table = Table(paragraphs, colWidths=widths, repeatRows=1, hAlign="LEFT")
    commands = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Arial-Bold"),
        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.35, GRID),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    for index in range(1, len(rows)):
        if index % 2 == 0:
            commands.append(("BACKGROUND", (0, index), (-1, index), PALE))
    table.setStyle(TableStyle(commands))
    return table


def parse_markdown(source: str, usable_width: float, style_map: dict[str, ParagraphStyle]) -> list[Flowable]:
    lines = source.splitlines()
    story: list[Flowable] = []
    # Title metadata is fixed at the head of the source.
    title = lines[0].removeprefix("# ").strip()
    author = lines[2].replace("**", "").strip()
    affiliation = lines[3].strip()
    version = lines[4].strip()
    story.extend([
        Spacer(1, 0.08 * inch),
        Paragraph(inline(title), style_map["title"]),
        Rule(usable_width),
        Spacer(1, 5),
        Paragraph(inline(author), style_map["author"]),
        Paragraph(inline(affiliation), style_map["meta"]),
        Paragraph(inline(version), style_map["meta"]),
        Spacer(1, 9),
    ])
    index = 5
    in_references = False
    abstract_next = False
    while index < len(lines):
        raw = lines[index]
        stripped = raw.strip()
        if not stripped:
            index += 1
            continue
        if stripped.startswith("<!--FIG:"):
            name = stripped.removeprefix("<!--FIG:").removesuffix("-->")
            maker, caption = FIGURES[name]
            story.append(KeepTogether([maker(usable_width), Paragraph(inline(caption), style_map["caption"])]))
            index += 1
            continue
        if stripped.startswith("## "):
            heading = stripped[3:]
            if heading.startswith("Appendix"):
                story.append(PageBreak())
            if heading == "References":
                in_references = True
            abstract_next = heading == "Abstract"
            story.append(Paragraph(inline(heading), style_map["h1"]))
            index += 1
            continue
        if stripped.startswith("### "):
            story.append(Paragraph(inline(stripped[4:]), style_map["h2"]))
            index += 1
            continue
        if stripped == "$$":
            equation_lines: list[str] = []
            index += 1
            while index < len(lines) and lines[index].strip() != "$$":
                equation_lines.append(lines[index])
                index += 1
            index += 1
            equation = Paragraph(equation_html(" ".join(equation_lines)), style_map["equation"])
            story.append(KeepTogether([Spacer(1, 2), equation, Spacer(1, 7)]))
            continue
        if stripped.startswith("```"):
            code_lines: list[str] = []
            index += 1
            while index < len(lines) and not lines[index].strip().startswith("```"):
                code_lines.append(lines[index])
                index += 1
            index += 1
            story.append(Paragraph("<br/>".join(html.escape(line) for line in code_lines), style_map["code"]))
            continue
        if stripped.startswith("|") and index + 1 < len(lines) and lines[index + 1].strip().startswith("|"):
            table_lines: list[str] = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                table_lines.append(lines[index])
                index += 1
            story.append(table_from_lines(table_lines, usable_width, style_map))
            story.append(Spacer(1, 7))
            continue
        if stripped.startswith("- ") or re.match(r"\d+\. ", stripped):
            while index < len(lines):
                item = lines[index].strip()
                match = re.match(r"(?:- |\d+\. )(.*)", item)
                if not match:
                    break
                story.append(Paragraph(inline(match.group(1)), style_map["bullet"], bulletText="•"))
                index += 1
            story.append(Spacer(1, 3))
            continue
        if stripped.startswith("**Table") and stripped.endswith("**"):
            story.append(Paragraph(inline(stripped[2:-2]), style_map["table_caption"]))
            index += 1
            continue

        paragraph_lines = [stripped]
        index += 1
        while index < len(lines):
            nxt = lines[index].strip()
            if not nxt or nxt.startswith(("## ", "### ", "$$", "```", "|", "- ", "<!--FIG:")) or re.match(r"\d+\. ", nxt):
                break
            if nxt.startswith("**Table"):
                break
            paragraph_lines.append(nxt)
            index += 1
        text = " ".join(paragraph_lines)
        style = style_map["reference"] if in_references else (style_map["abstract"] if abstract_next else style_map["body"])
        story.append(Paragraph(inline(text), style))
        abstract_next = False
    return story


class PreprintDocTemplate(BaseDocTemplate):
    def __init__(self, filename: str, **kwargs):
        super().__init__(filename, **kwargs)
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="body")
        # Draw furniture after flowables so split tables and figures cannot paint over it.
        self.addPageTemplates(PageTemplate(id="paper", frames=[frame], onPageEnd=self._page))

    @staticmethod
    def _page(canvas, doc) -> None:
        canvas.saveState()
        canvas.setTitle("FE-E: Finite-Element and Entropy Control of Adjoint Propagation in Deep Transformers")
        canvas.setAuthor("XUEZHENG WANG")
        canvas.setSubject("FE-E preprint and reproducibility report")
        page = canvas.getPageNumber()
        width, height = letter
        if page > 1:
            canvas.setStrokeColor(GRID)
            canvas.setLineWidth(0.5)
            canvas.line(doc.leftMargin, height - 0.48 * inch, width - doc.rightMargin, height - 0.48 * inch)
            canvas.setFont("Arial", 7.2)
            canvas.setFillColor(MUTED)
            canvas.drawString(doc.leftMargin, height - 0.37 * inch, "FE-E: Finite-Element and Entropy Control of Adjoint Propagation")
            canvas.drawRightString(width - doc.rightMargin, height - 0.37 * inch, "XUEZHENG WANG · PREPRINT")
        canvas.setStrokeColor(GRID)
        canvas.line(doc.leftMargin, 0.43 * inch, width - doc.rightMargin, 0.43 * inch)
        canvas.setFont("Arial", 7.2)
        canvas.setFillColor(MUTED)
        canvas.drawString(doc.leftMargin, 0.27 * inch, "Version 0.1 · August 2026")
        canvas.drawRightString(width - doc.rightMargin, 0.27 * inch, str(page))
        canvas.restoreState()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=PAPER)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    register_fonts()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    left = right = 0.68 * inch
    top, bottom = 0.78 * inch, 0.56 * inch
    doc = PreprintDocTemplate(
        str(args.output), pagesize=letter, leftMargin=left, rightMargin=right,
        topMargin=top, bottomMargin=bottom,
    )
    style_map = styles()
    story = parse_markdown(args.source.read_text(), doc.width, style_map)
    doc.build(story)
    print(args.output)


if __name__ == "__main__":
    main()
