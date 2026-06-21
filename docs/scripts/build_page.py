#!/usr/bin/env python3
"""Build the self-contained RAHA project page and every figure shown on it.

The script intentionally uses only the Python standard library for charts and diagrams.
For paper qualitative crops, it calls local Ghostscript and ImageMagick when available,
because the source is the local RAHA_final.pdf.
"""
from __future__ import annotations

import base64
import html
import math
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "generated_figures"
PDF = ROOT / "RAHA_final.pdf"
FONT = ROOT / "assets" / "PretendardVariable.woff2"
KAIST_LOGO = ROOT / "assets" / "kaist_logo_official_trim.png"
ETRI_LOGO = ROOT / "assets" / "etri_wordmark_official_trim.png"
MATCHA_ICON = ROOT / "assets" / "matcha_leaf.svg"

BASELINE = "Baseline"
OURS = "RAHA"
COLORS = {
    BASELINE: "#d49a8c",
    OURS: "#76a99a",
    "ink": "#000000",
    "muted": "#000000",
    "paper": "#ffffff",
    "panel": "#ffffff",
    "line": "#c5d9e8",
    "blue": "#9fb7d9",
    "yellow": "#ead48f",
    "violet": "#c3b2dc",
    "lime": "#bfd4a7",
}
DATASET_CLASSES = {"Flickr8k": "dataset-flickr8k", "Flickr30k": "dataset-flickr30k", "COCO": "dataset-coco"}
DATASET_ACCENTS = {
    "Flickr8k": ("#6f9fd0", "#eaf3fb"),
    "Flickr30k": ("#76a99a", "#e9f3ef"),
    "COCO": ("#d49a8c", "#f8ebe7"),
}
PAIR_CLASSES = {100: "pair-100", 200: "pair-200", 500: "pair-500", 1000: "pair-1000"}
PAIR_ACCENTS = {
    100: ("#c3b2dc", "#f2eef8"),
    200: ("#9fb7d9", "#edf4fb"),
    500: ("#ead48f", "#fbf6df"),
    1000: ("#bfd4a7", "#f0f6ea"),
}
SVG_TEXT_STYLE = '<style>text{font-family:Pretendard;letter-spacing:0}</style>'

# Mean retrieval scores. The paper's asterisked batch-64 entries are folded into
# the same Baseline label per page owner instruction.
RETRIEVAL: Dict[str, Dict[str, Dict[int, float]]] = {
    "Flickr8k": {
        BASELINE: {100: 20.4, 200: 19.5, 500: 25.9, 1000: 30.6},
        OURS: {100: 20.4, 200: 25.3, 500: 30.7, 1000: 37.1},
    },
    "Flickr30k": {
        BASELINE: {100: 22.8, 200: 22.0, 500: 28.9, 1000: 28.4},
        OURS: {100: 20.7, 200: 25.7, 500: 32.9, 1000: 38.0},
    },
    "COCO": {
        BASELINE: {100: 7.0, 200: 8.3, 500: 11.2, 1000: 14.2},
        OURS: {100: 7.2, 200: 10.2, 500: 13.7, 1000: 18.6},
    },
}

TRANSFER = {
    100: {BASELINE: 7.3, OURS: 6.9, "source_baseline": 20.4, "source_raha": 20.4},
    200: {BASELINE: 7.2, OURS: 8.7, "source_baseline": 18.8, "source_raha": 25.3},
    500: {BASELINE: 8.7, OURS: 12.7, "source_baseline": 25.9, "source_raha": 30.7},
}

CLASSIFICATION = {
    "CIFAR-100": {BASELINE: 14.55, OURS: 15.60},
    "CUB-200": {BASELINE: 12.53, OURS: 14.43},
    "Cars": {BASELINE: 5.63, OURS: 7.80},
    "ImageNet-1K": {BASELINE: 7.62, OURS: 7.88},
}

APPENDIX_CROPS = [
    ("cifar", "CIFAR-100", "840x256+270+226"),
    ("cub", "CUB-200-2011", "840x216+270+570"),
    ("cars", "Stanford Cars", "840x218+270+848"),
    ("imagenet", "ImageNet-1K", "840x228+270+1124"),
]


def esc(text: object) -> str:
    return html.escape(str(text), quote=True)


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def font_face_css() -> str:
    if not FONT.exists():
        return ""
    encoded = base64.b64encode(FONT.read_bytes()).decode("ascii")
    return (
        '@font-face{font-family:"Pretendard";font-style:normal;font-weight:100 900;'
        'font-display:swap;src:url("data:font/woff2;base64,' + encoded + '") format("woff2-variations");}'
    )

def data_uri(path: Path, mime: str) -> str:
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def svg_uri(svg: str) -> str:
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("ascii")


FIGURE_FONT_SCALE = 1.12


def scaled_figure_svg(svg: str) -> str:
    def repl(match: re.Match[str]) -> str:
        size = float(match.group(1)) * FIGURE_FONT_SCALE
        text = f"{size:.1f}".rstrip("0").rstrip(".")
        return f'font-size="{text}"'

    return re.sub(r'font-size="([0-9]+(?:\.[0-9]+)?)"', repl, svg)


def raha_mark_svg() -> str:
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" role="img">'
        '<circle cx="32" cy="32" r="29" fill="#e7f0e9" stroke="#cbdccd" stroke-width="3"/>'
        '<circle cx="32" cy="32" r="20" fill="none" stroke="#ffffff" stroke-width="8"/>'
        '<text x="32" y="41" text-anchor="middle" font-family="Pretendard" font-size="30" font-weight="900" fill="#000000">R</text>'
        '</svg>'
    )


def svg_inner(path: Path) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8")
    svg_start = text.find("<svg")
    body_start = text.find(">", svg_start)
    body_end = text.rfind("</svg>")
    if svg_start < 0 or body_start < 0 or body_end < 0:
        return ""
    return text[body_start + 1:body_end].strip()


def polyline(points: Iterable[Tuple[float, float]]) -> str:
    pts = list(points)
    if not pts:
        return ""
    first, *rest = pts
    return "M" + f"{first[0]:.1f},{first[1]:.1f}" + "".join(f" L{x:.1f},{y:.1f}" for x, y in rest)


def retrieval_svg() -> str:
    budgets = [100, 200, 500, 1000]
    width, height = 860, 940
    panel_w, panel_h = 760, 218
    left, top, gap = 58, 154, 238
    parts: List[str] = []

    def axis_ceiling(dataset: str) -> int:
        highest = max(max(series.values()) for series in RETRIEVAL[dataset].values())
        step = 5 if highest <= 25 else 10
        return int(math.ceil(highest / step) * step)

    def axis_ticks(max_y: int) -> List[int]:
        step = 5 if max_y <= 25 else 10
        return list(range(0, max_y + 1, step))

    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img">')
    parts.append(SVG_TEXT_STYLE)
    parts.append(f'<rect width="{width}" height="{height}" rx="26" fill="#ffffff"/>')
    parts.append('<path d="M42 74H818" stroke="#1d76bb" stroke-width="3"/>')
    parts.append('<text x="42" y="52" font-size="31" font-weight="900" fill="#000000">Retrieval across synthetic budgets</text>')
    parts.append('<text x="42" y="101" font-size="18" fill="#000000">Same protocol with per-dataset y-axes. Higher is better.</text>')
    parts.append(f'<rect x="306" y="112" width="248" height="36" rx="16" fill="#ffffff" stroke="#c5d9e8" opacity=".94"/>')
    parts.append(f'<circle cx="332" cy="130" r="8" fill="{COLORS[BASELINE]}"/><text x="350" y="136" font-size="17" font-weight="840" fill="#000000">Baseline</text>')
    parts.append(f'<circle cx="454" cy="130" r="8" fill="{COLORS[OURS]}"/><text x="472" y="136" font-size="17" font-weight="840" fill="#000000">RAHA</text>')

    for i, (dataset, series) in enumerate(RETRIEVAL.items()):
        y0 = top + i * gap
        dataset_color, dataset_soft = DATASET_ACCENTS[dataset]
        max_y = axis_ceiling(dataset)
        parts.append(f'<g transform="translate({left},{y0})">')
        parts.append(f'<rect x="0" y="0" width="{panel_w}" height="{panel_h}" rx="10" fill="#ffffff" stroke="{dataset_color}" stroke-width="1.6"/>')
        parts.append(f'<rect x="0" y="0" width="10" height="{panel_h}" rx="5" fill="{dataset_color}" opacity=".86"/>')
        parts.append(f'<rect x="20" y="14" width="144" height="30" rx="15" fill="{dataset_soft}" stroke="{dataset_color}"/>')
        parts.append(f'<text x="92" y="35" text-anchor="middle" font-size="24" font-weight="900" fill="#000000">{esc(dataset)}</text>')
        chart_x, chart_y, chart_w, chart_h = 72, 64, 640, 106
        parts.append(f'<line x1="{chart_x}" y1="{chart_y}" x2="{chart_x}" y2="{chart_y+chart_h}" stroke="#9fb7d9" stroke-width="1.5"/>')
        parts.append(f'<line x1="{chart_x}" y1="{chart_y+chart_h}" x2="{chart_x+chart_w}" y2="{chart_y+chart_h}" stroke="#9fb7d9" stroke-width="1.5"/>')
        for tick in axis_ticks(max_y):
            y = chart_y + chart_h - tick / max_y * chart_h
            parts.append(f'<line x1="{chart_x}" y1="{y:.1f}" x2="{chart_x+chart_w}" y2="{y:.1f}" stroke="#d6ebf8"/>')
            parts.append(f'<text x="{chart_x-18}" y="{y+6:.1f}" text-anchor="end" font-size="18" font-weight="760" fill="#000000">{tick}</text>')
        x_positions = {b: chart_x + j * (chart_w / (len(budgets) - 1)) for j, b in enumerate(budgets)}
        for b in budgets:
            pair_color, _pair_soft = PAIR_ACCENTS[b]
            x = x_positions[b]
            parts.append(f'<text x="{x:.1f}" y="{chart_y+chart_h+31}" text-anchor="middle" font-size="16" font-weight="850" fill="#000000">{b}</text>')
            parts.append(f'<rect x="{x-25:.1f}" y="{chart_y+chart_h+39}" width="50" height="6" rx="3" fill="{pair_color}" opacity=".95"/>')
        for name in [BASELINE, OURS]:
            vals = series[name]
            pts = [(x_positions[b], chart_y + chart_h - vals[b] / max_y * chart_h) for b in budgets]
            parts.append(f'<path d="{polyline(pts)}" fill="none" stroke="{COLORS[name]}" stroke-width="5" stroke-linecap="round" stroke-linejoin="round"/>')
            for b, (x, y) in zip(budgets, pts):
                pair_color, _pair_soft = PAIR_ACCENTS[b]
                parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="{pair_color}" stroke="{COLORS[name]}" stroke-width="4.5"/>')
                offset = -12 if name == OURS else 24
                parts.append(f'<text x="{x:.1f}" y="{y+offset:.1f}" text-anchor="middle" font-size="16" font-weight="900" fill="#000000">{vals[b]:.1f}</text>')
        parts.append('</g>')
    parts.append('<text x="430" y="912" text-anchor="middle" font-size="17" fill="#000000">Mean of image-retrieval and text-retrieval scores. Pair colors match the table.</text>')
    parts.append('</svg>')
    return scaled_figure_svg("".join(parts))

def transfer_svg() -> str:
    width, height = 860, 430
    budgets = [100, 200, 500]
    max_y = 15
    major_ticks = [0, 5, 10, 15]
    parts: List[str] = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img">']
    parts.append(SVG_TEXT_STYLE)
    parts.append(f'<rect width="{width}" height="{height}" rx="24" fill="#ffffff"/>')
    parts.append(f'<rect x="18" y="18" width="{width-36}" height="{height-36}" rx="18" fill="#fbfdff" stroke="#c5d9e8"/>')
    parts.append('<text x="40" y="58" font-size="33" font-weight="900" fill="#000000">Cross-architecture transfer</text>')
    parts.append('<text x="40" y="91" font-size="18" fill="#000000">Flickr8k mean under replacement backbones and text encoders.</text>')
    chart_x, chart_y, chart_w, chart_h = 74, 144, 732, 210
    for tick in major_ticks:
        y = chart_y + chart_h - tick / max_y * chart_h
        stroke = "#c5d9e8" if tick in (0, max_y) else "#d6ebf8"
        line_w = "1.7" if tick in (0, max_y) else "1.2"
        parts.append(f'<line x1="{chart_x}" y1="{y:.1f}" x2="{chart_x+chart_w}" y2="{y:.1f}" stroke="{stroke}" stroke-width="{line_w}"/>')
        parts.append(f'<text x="{chart_x-16}" y="{y+6:.1f}" text-anchor="end" font-size="18" font-weight="760" fill="#000000">{tick}</text>')
    parts.append(f'<path d="M{chart_x} {chart_y}V{chart_y+chart_h}H{chart_x+chart_w}" fill="none" stroke="#9fb7d9" stroke-width="1.8"/>')
    group_w = chart_w / len(budgets)
    bar_w = 62
    for i, b in enumerate(budgets):
        cx = chart_x + group_w * i + group_w / 2
        for j, name in enumerate([BASELINE, OURS]):
            val = TRANSFER[b][name]
            h = val / max_y * chart_h
            x = cx + (-bar_w - 9 if j == 0 else 9)
            y = chart_y + chart_h - h
            fill = COLORS[name]
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w}" height="{h:.1f}" rx="8" fill="{fill}"/>')
            parts.append(f'<text x="{x+bar_w/2:.1f}" y="{y-12:.1f}" text-anchor="middle" font-size="20" font-weight="900" fill="#000000">{val:.1f}</text>')
        parts.append(f'<text x="{cx:.1f}" y="{chart_y+chart_h+36}" text-anchor="middle" font-size="20" font-weight="900" fill="#000000">{b}</text>')
    parts.append(f'<rect x="40" y="106" width="246" height="31" rx="15.5" fill="#fbfdff" stroke="#c5d9e8" opacity=".98"/>')
    parts.append(f'<rect x="58" y="115" width="14" height="14" rx="3" fill="{COLORS[BASELINE]}"/><text x="82" y="128" font-size="17" font-weight="840" fill="#000000">Baseline</text>')
    parts.append(f'<rect x="176" y="115" width="14" height="14" rx="3" fill="{COLORS[OURS]}"/><text x="200" y="128" font-size="17" font-weight="840" fill="#000000">RAHA</text>')
    parts.append('<path d="M64 386H804" stroke="#c5d9e8" stroke-width="2"/>')
    parts.append('<text x="64" y="413" font-size="17" fill="#000000">Higher is better. Transfer improves as synthetic image-text capacity grows.</text>')
    parts.append('</svg>')
    return scaled_figure_svg("".join(parts))

def classification_svg() -> str:
    width, height = 860, 450
    max_y = 18
    names = list(CLASSIFICATION.keys())
    parts: List[str] = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img">']
    parts.append(SVG_TEXT_STYLE)
    parts.append(f'<rect width="{width}" height="{height}" rx="26" fill="#ffffff"/>')
    parts.append('<text x="42" y="58" font-size="31" font-weight="900" fill="#000000">Prompted image classification at 100 pairs</text>')
    parts.append('<text x="42" y="91" font-size="18" fill="#000000">Top-1 accuracy under CLIP-style prompting on four recognition benchmarks.</text>')
    chart_x, chart_y, chart_w, chart_h = 78, 124, 710, 220
    for tick in [0, 5, 10, 15]:
        y = chart_y + chart_h - tick / max_y * chart_h
        parts.append(f'<line x1="{chart_x}" y1="{y:.1f}" x2="{chart_x+chart_w}" y2="{y:.1f}" stroke="#d6ebf8"/>')
        parts.append(f'<text x="{chart_x-18}" y="{y+6:.1f}" text-anchor="end" font-size="18" font-weight="760" fill="#000000">{tick}</text>')
    group_w = chart_w / len(names)
    bar_w = 50
    for i, name in enumerate(names):
        cx = chart_x + group_w * i + group_w / 2
        for j, method in enumerate([BASELINE, OURS]):
            val = CLASSIFICATION[name][method]
            h = val / max_y * chart_h
            x = cx + (-bar_w - 8 if j == 0 else 8)
            y = chart_y + chart_h - h
            fill = COLORS[method]
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w}" height="{h:.1f}" rx="7" fill="{fill}"/>')
            parts.append(f'<text x="{x+bar_w/2:.1f}" y="{y-10:.1f}" text-anchor="middle" font-size="15" font-weight="900" fill="#000000">{val:.2f}</text>')
        parts.append(f'<text x="{cx:.1f}" y="{chart_y+chart_h+38}" text-anchor="middle" font-size="16" font-weight="900" fill="#000000">{esc(name)}</text>')
    parts.append(f'<rect x="318" y="405" width="224" height="34" rx="16" fill="#ffffff" stroke="#c5d9e8" opacity=".94"/>')
    parts.append(f'<rect x="342" y="415" width="15" height="15" rx="3" fill="{COLORS[BASELINE]}"/><text x="366" y="429" font-size="17" font-weight="840" fill="#000000">Baseline</text>')
    parts.append(f'<rect x="452" y="415" width="15" height="15" rx="3" fill="{COLORS[OURS]}"/><text x="476" y="429" font-size="17" font-weight="840" fill="#000000">RAHA</text>')
    parts.append('</svg>')
    return scaled_figure_svg("".join(parts))

def semantic_svg() -> str:
    parts: List[str] = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1180 690" role="img">']
    parts.append(SVG_TEXT_STYLE)

    def pill(x: float, y: float, text: str, fill: str = "#ffffff", stroke: str = "#c5d9e8", anchor: str = "middle", color: str = "#000000", weight: int = 850, size: int = 14) -> None:
        width = max(54, len(text) * (size * FIGURE_FONT_SCALE * 0.58) + 22)
        height = 26
        x0 = x - width / 2 if anchor == "middle" else x
        tx = x if anchor == "middle" else x + 11
        text_anchor = "middle" if anchor == "middle" else "start"
        parts.append(f'<rect x="{x0:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height}" rx="13" fill="{fill}" stroke="{stroke}"/>')
        parts.append(f'<text x="{tx:.1f}" y="{y+17:.1f}" text-anchor="{text_anchor}" font-size="{size}" font-weight="{weight}" fill="#000000">{esc(text)}</text>')

    def node(x: float, y: float, kind: str, fill: str) -> None:
        if kind == "I":
            parts.append(f'<rect x="{x-15:.1f}" y="{y-15:.1f}" width="30" height="30" rx="8" fill="{fill}" stroke="#ffffff" stroke-width="4"/>')
        else:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="16" fill="{fill}" stroke="#ffffff" stroke-width="4"/>')
        parts.append(f'<text x="{x:.1f}" y="{y+4:.1f}" text-anchor="middle" font-size="13" font-weight="900" fill="#000000">{kind}</text>')

    parts.append('<defs>')
    parts.append('<linearGradient id="sphereFill" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#ffffff"/><stop offset="1" stop-color="#eef3f8"/></linearGradient>')
    parts.append('<linearGradient id="hyperFill" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#eef3f8"/><stop offset="0.52" stop-color="#e7f0e9"/><stop offset="1" stop-color="#ffffff"/></linearGradient>')
    parts.append('<linearGradient id="rangeLine" x1="0" x2="1"><stop offset="0" stop-color="#78a99a"/><stop offset="1" stop-color="#9fb7d9"/></linearGradient>')
    parts.append('<linearGradient id="dogCone" x1="0" x2="1"><stop offset="0" stop-color="#d49a8c" stop-opacity="0.36"/><stop offset="1" stop-color="#d49a8c" stop-opacity="0.05"/></linearGradient>')
    parts.append('<linearGradient id="catCone" x1="0" x2="1"><stop offset="0" stop-color="#c3b2dc" stop-opacity="0.36"/><stop offset="1" stop-color="#c3b2dc" stop-opacity="0.05"/></linearGradient>')
    parts.append('<marker id="softArrow" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#1d76bb"/></marker>')
    parts.append('</defs>')
    parts.append('<rect width="1180" height="690" rx="30" fill="#ffffff"/>')
    parts.append('<rect x="26" y="26" width="1128" height="638" rx="24" fill="#ffffff" stroke="#c5d9e8"/>')
    parts.append('<text x="590" y="64" text-anchor="middle" font-size="31" font-weight="900" fill="#000000">Hypersphere matching vs. hyperboloid alignment</text>')
    parts.append('<text x="590" y="94" text-anchor="middle" font-size="17" fill="#000000">Hyperbolic geometry keeps generic text near many descendants.</text>')
    parts.append('<text x="590" y="118" text-anchor="middle" font-size="17" fill="#000000">Specific image-text instances stay separable in detailed regions.</text>')

    # Left panel: Euclidean hypersphere.
    parts.append('<g transform="translate(36,134)">')
    parts.append('<rect x="0" y="0" width="438" height="456" rx="18" fill="#f7f1e8" stroke="#c5d9e8"/>')
    parts.append('<text x="219" y="42" text-anchor="middle" font-size="20" font-weight="900" fill="#000000">Euclidean hypersphere</text>')
    parts.append('<text x="219" y="66" text-anchor="middle" font-size="15" fill="#000000">unit-normalized embeddings</text>')
    parts.append('<text x="219" y="86" text-anchor="middle" font-size="15" fill="#000000">share one angular metric</text>')
    parts.append('<circle cx="219" cy="230" r="128" fill="url(#sphereFill)" stroke="#c5d9e8" stroke-width="2"/>')
    parts.append('<ellipse cx="219" cy="230" rx="128" ry="38" fill="none" stroke="#d8ccbe" stroke-width="2"/>')
    parts.append('<ellipse cx="219" cy="230" rx="46" ry="128" fill="none" stroke="#e1d6ca" stroke-width="2"/>')
    parts.append('<path d="M112 160C172 204 266 256 326 300" fill="none" stroke="#e1d6ca" stroke-width="2"/>')
    parts.append('<path d="M326 160C266 204 172 256 112 300" fill="none" stroke="#e1d6ca" stroke-width="2"/>')
    parts.append('<path d="M138 146C196 218 250 252 308 314" fill="none" stroke="#bbaea0" stroke-width="2.4" stroke-dasharray="7 8"/>')
    parts.append('<path d="M302 146C262 202 244 228 224 244" fill="none" stroke="#bbaea0" stroke-width="2.4" stroke-dasharray="7 8"/>')
    parts.append('<path d="M136 318C172 286 193 266 212 248" fill="none" stroke="#bbaea0" stroke-width="2.4" stroke-dasharray="7 8"/>')
    for x, y, k, fill in [(138,146,"T","#9fb7d9"),(302,146,"T","#9fb7d9"),(136,318,"I","#78a99a"),(300,314,"I","#78a99a"),(212,248,"I","#78a99a"),(224,244,"T","#9fb7d9")]:
        node(x, y, k, fill)
    pill(138, 110, "text: dog", "#eef3f8", "#cfdae8")
    pill(302, 110, "text: cat", "#eef3f8", "#cfdae8")
    pill(136, 342, "dog image", "#e7f0e9", "#cbdccd")
    pill(300, 338, "cat image", "#e7f0e9", "#cbdccd")
    pill(218, 274, "dog+cat midpoint", "#f6e7dc", "#e6cfc1")
    parts.append('<rect x="34" y="390" width="370" height="42" rx="14" fill="#f6e7dc" stroke="#e6cfc1"/>')
    parts.append('<text x="219" y="407" text-anchor="middle" font-size="15" font-weight="900" fill="#000000">Flat semantic average</text>')
    parts.append('<text x="219" y="424" text-anchor="middle" font-size="14" fill="#000000">the mixed pair is forced toward a spherical midpoint</text>')
    parts.append('</g>')

    # Middle arrow and MATCHA lift.
    matcha_icon = svg_inner(MATCHA_ICON)
    parts.append('<g transform="translate(506,198)">')
    parts.append('<path d="M-26 132H-6" stroke="#1d76bb" stroke-width="3.2" stroke-linecap="round" marker-end="url(#softArrow)"/>')
    parts.append('<path d="M150 132H166" stroke="#1d76bb" stroke-width="3.2" stroke-linecap="round" marker-end="url(#softArrow)"/>')
    parts.append('<rect x="0" y="46" width="144" height="174" rx="20" fill="#ffffff" stroke="#a7d6f3" opacity=".92"/>')
    parts.append('<text x="72" y="77" text-anchor="middle" font-size="11.6" fill="#000000"><tspan font-weight="950">Match</tspan> <tspan font-weight="950">H</tspan>yperbolically</text>')
    parts.append('<text x="72" y="94" text-anchor="middle" font-size="11.6" fill="#000000"><tspan font-weight="950">A</tspan>ligned Distributions</text>')
    parts.append('<circle cx="72" cy="135" r="31" fill="#ffffff" stroke="#cbdccd"/>')
    if matcha_icon:
        parts.append(f'<g transform="translate(42,105) scale(0.1171875)">{matcha_icon}</g>')
    else:
        parts.append('<path d="M44 138C56 112 78 112 100 128C80 166 58 164 44 138Z" fill="#8cd36f" stroke="#4caf28" stroke-width="2"/>')
    parts.append('<text x="72" y="172" text-anchor="middle" font-size="11.5" font-weight="950" fill="#000000">MATCHA</text>')
    pill(72, 186, "Lorentz model", "#eef3f8", "#cfdae8")
    parts.append('</g>')

    # Right panel: Lorentz hyperboloid.
    parts.append('<g transform="translate(674,134)">')
    parts.append('<rect x="0" y="0" width="470" height="456" rx="18" fill="#f7f1e8" stroke="#c5d9e8"/>')
    parts.append('<text x="235" y="42" text-anchor="middle" font-size="20" font-weight="900" fill="#000000">Hyperbolic space: hyperboloid</text>')
    parts.append('<text x="235" y="66" text-anchor="middle" font-size="15" fill="#000000">generic text stays near the funnel tip</text>')
    parts.append('<text x="235" y="86" text-anchor="middle" font-size="15" fill="#000000">specific image-text pairs spread into details</text>')
    parts.append('<path d="M235 95V382" stroke="#c8baad" stroke-width="2" stroke-dasharray="6 7"/>')
    parts.append('<text x="266" y="375" font-size="14" font-weight="900" fill="#000000">time-like axis</text>')
    parts.append('<path d="M136 112C160 150 178 195 194 242C207 282 218 322 235 366C252 322 263 282 276 242C292 195 310 150 334 112" fill="url(#hyperFill)" stroke="#c5d9e8" stroke-width="2" opacity=".84"/>')
    for cy, rx, ry in [(124,98,22),(168,78,18),(212,58,14),(256,40,10),(306,24,7),(350,10,4)]:
        parts.append(f'<ellipse cx="235" cy="{cy}" rx="{rx}" ry="{ry}" fill="none" stroke="#d9cec1" stroke-width="2"/>')
    parts.append('<path d="M235 350C220 300 208 250 194 198C184 161 172 136 156 112" fill="none" stroke="#e5dad0" stroke-width="2"/>')
    parts.append('<path d="M235 350C250 300 262 250 276 198C286 161 298 136 314 112" fill="none" stroke="#e5dad0" stroke-width="2"/>')
    parts.append('<path d="M235 346C252 300 278 247 302 195C320 157 337 133 355 112" fill="none" stroke="url(#rangeLine)" stroke-width="16" stroke-linecap="round" opacity=".74"/>')
    parts.append('<path d="M142 124C200 106 270 106 344 124" fill="none" stroke="#d49a8c" stroke-width="7" stroke-linecap="round" stroke-dasharray="2 17" opacity=".56"/>')
    parts.append('<circle cx="235" cy="350" r="22" fill="#f2dfaa" stroke="#ffffff" stroke-width="5"/>')
    parts.append('<text x="235" y="346" text-anchor="middle" font-size="14" font-weight="900" fill="#000000">generic text</text>')
    parts.append('<text x="235" y="362" text-anchor="middle" font-size="13" fill="#000000">"a dog"</text>')
    for x, y, k, fill in [(300,206,"T","#9fb7d9"),(322,186,"I","#78a99a"),(178,206,"T","#9fb7d9"),(156,186,"I","#78a99a"),(334,132,"T","#9fb7d9"),(358,114,"I","#78a99a")]:
        node(x, y, k, fill)
    parts.append('<path d="M300 206L322 186M178 206L156 186M334 132L358 114" stroke="#8f8378" stroke-width="2.2" stroke-linecap="round"/>')
    pill(354, 214, "dog pair", "#e7f0e9", "#cbdccd")
    pill(46, 214, "cat pair", "#e7f0e9", "#cbdccd", anchor="start")
    pill(292, 76, "dog + cat pair", "#eef3f8", "#cfdae8")

    # Projected entailment-cone inset, mirroring HYPE Fig. 2 without copying it.
    parts.append('<g transform="translate(24,292)" opacity=".82">')
    parts.append('<rect x="0" y="0" width="150" height="116" rx="14" fill="#ffffff" stroke="#c5d9e8"/>')
    parts.append('<text x="75" y="22" text-anchor="middle" font-size="14" font-weight="900" fill="#000000">projection view</text>')
    parts.append('<circle cx="75" cy="68" r="36" fill="#ffffff" stroke="#c5d9e8"/>')
    parts.append('<path d="M75 68 L37 48 A48 48 0 0 0 52 98 Z" fill="url(#catCone)"/>')
    parts.append('<path d="M75 68 L113 48 A48 48 0 0 1 98 98 Z" fill="url(#dogCone)"/>')
    parts.append('<circle cx="54" cy="62" r="5" fill="#9fb7d9"/><text x="42" y="48" font-size="12" font-weight="800" fill="#000000">cat</text>')
    parts.append('<circle cx="96" cy="62" r="5" fill="#9fb7d9"/><text x="100" y="48" font-size="12" font-weight="800" fill="#000000">dog</text>')
    parts.append('<rect x="70" y="80" width="10" height="10" rx="3" fill="#78a99a"/><text x="84" y="91" font-size="12" font-weight="800" fill="#000000">overlap</text>')
    parts.append('</g>')
    pill(120, 424, "shared semantic range", "#e7f0e9", "#cbdccd")
    pill(336, 424, "controlled residual detail", "#f6e7dc", "#e6cfc1")
    parts.append('</g>')

    parts.append('<rect x="58" y="596" width="440" height="42" rx="14" fill="#ffffff" stroke="#c5d9e8"/>')
    parts.append('<rect x="82" y="610" width="14" height="14" rx="4" fill="#78a99a"/><text x="104" y="622" font-size="14" font-weight="800" fill="#000000">image embedding</text>')
    parts.append('<circle cx="300" cy="617" r="7" fill="#9fb7d9"/><text x="318" y="622" font-size="14" font-weight="800" fill="#000000">text embedding</text>')
    parts.append('<rect x="528" y="596" width="570" height="42" rx="14" fill="#e7f0e9" stroke="#cbdccd"/>')
    parts.append('<text x="813" y="622" text-anchor="middle" font-size="14" font-weight="900" fill="#000000">Generic text stays broad while image-text details stay separable.</text>')
    parts.append('</svg>')
    return scaled_figure_svg("".join(parts))

def run(cmd: List[str]) -> bool:
    try:
        subprocess.run(cmd, cwd=ROOT, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except Exception:
        return False


def render_page(page: int, out: Path) -> bool:
    if not PDF.exists() or not shutil.which("gs"):
        return False
    return run([
        "gs", "-dNOSAFER", "-dNOPAUSE", "-dBATCH", "-sDEVICE=pngalpha", "-r160",
        f"-dFirstPage={page}", f"-dLastPage={page}", f"-sOutputFile={out}", str(PDF)
    ])


def crop(src: Path, geometry: str, out: Path) -> bool:
    if not src.exists() or not shutil.which("convert"):
        return False
    return run(["convert", str(src), "-crop", geometry, "+repage", "-fuzz", "2%", "-trim", "+repage", "-strip", str(out)])


def make_png_crops() -> Dict[str, Path]:
    page12 = GEN / "page_12.png"
    page14 = GEN / "page_14.png"
    page26 = GEN / "page_26.png"
    page30 = GEN / "page_30.png"
    render_page(12, page12)
    render_page(14, page14)
    render_page(26, page26)
    render_page(30, page30)

    if not page12.exists() and Path("/tmp/raha_page_12.png").exists():
        page12.write_bytes(Path("/tmp/raha_page_12.png").read_bytes())
    if not page14.exists() and Path("/tmp/raha_page_14.png").exists():
        page14.write_bytes(Path("/tmp/raha_page_14.png").read_bytes())
    if not page26.exists() and Path("/tmp/raha_app_01.png").exists():
        page26.write_bytes(Path("/tmp/raha_app_01.png").read_bytes())

    out = {}
    ablation = GEN / "fig2_ablation.png"
    if crop(page14, "832x244+264+234", ablation):
        out["ablation"] = ablation
    fig_a6 = GEN / "fig_a6_sensitivity.png"
    if crop(page30, "832x205+264+241", fig_a6):
        out["fig_a6"] = fig_a6
    main = GEN / "main_qualitative.png"
    if crop(page12, "785x282+292+606", main):
        out["main_qual"] = main
    for key, _label, geometry in APPENDIX_CROPS:
        target = GEN / f"qual_{key}.png"
        if crop(page26, geometry, target):
            out[f"qual_{key}"] = target
    return out


def make_tables() -> Tuple[str, str, str]:
    retrieval_rows = []
    for dataset, series in RETRIEVAL.items():
        dataset_class = DATASET_CLASSES[dataset]
        for budget in [100, 200, 500, 1000]:
            pair_class = PAIR_CLASSES[budget]
            b = series[BASELINE][budget]
            r = series[OURS][budget]
            delta = r - b
            sign = 'pos' if delta >= 0 else 'neg'
            retrieval_rows.append(
                f"<tr class=\"{dataset_class} {pair_class}\"><td><span class=\"dataset-pill {dataset_class}\">{esc(dataset)}</span></td><td><span class=\"pair-pill {pair_class}\">{budget}</span></td><td>{b:.1f}</td><td>{r:.1f}</td><td class=\"delta {sign}\">{delta:+.1f}</td></tr>"
            )
    retrieval_table = "".join(retrieval_rows)

    transfer_rows = []
    for budget in [100, 200, 500]:
        b = TRANSFER[budget][BASELINE]
        r = TRANSFER[budget][OURS]
        sign = 'pos' if r-b >= 0 else 'neg'
        transfer_rows.append(
            f"<tr><td>{budget}</td><td>{TRANSFER[budget]['source_baseline']:.1f}</td><td>{TRANSFER[budget]['source_raha']:.1f}</td><td>{b:.1f}</td><td>{r:.1f}</td><td class=\"delta {sign}\">{r-b:+.1f}</td></tr>"
        )
    transfer_table = "".join(transfer_rows)

    cls_rows = []
    for dataset, vals in CLASSIFICATION.items():
        delta = vals[OURS] - vals[BASELINE]
        sign = 'pos' if delta >= 0 else 'neg'
        cls_rows.append(
            f"<tr><td>{esc(dataset)}</td><td>{vals[BASELINE]:.2f}</td><td>{vals[OURS]:.2f}</td><td class=\"delta {sign}\">{delta:+.2f}</td></tr>"
        )
    cls_table = "".join(cls_rows)
    return retrieval_table, transfer_table, cls_table

def motivation_svg() -> str:
    parts: List[str] = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 940 790" role="img">']
    parts.append(SVG_TEXT_STYLE)

    def pill(x: float, y: float, text: str, fill: str, stroke: str, color: str = "#000000", size: int = 14, weight: int = 850) -> None:
        width = max(84, len(text) * (size * FIGURE_FONT_SCALE * 0.58) + 28)
        parts.append(f'<rect x="{x-width/2:.1f}" y="{y:.1f}" width="{width:.1f}" height="34" rx="17" fill="{fill}" stroke="{stroke}"/>')
        parts.append(f'<text x="{x:.1f}" y="{y+22:.1f}" text-anchor="middle" font-size="{size}" font-weight="{weight}" fill="#000000">{esc(text)}</text>')

    parts.append('<defs>')
    parts.append('<marker id="arrowMot" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="9" markerHeight="9" orient="auto"><path d="M0 0L10 5L0 10Z" fill="#1d76bb"/></marker>')
    parts.append('<linearGradient id="rankGrad" x1="0" x2="1"><stop offset="0" stop-color="#e7f0e9"/><stop offset="1" stop-color="#eef3f8"/></linearGradient>')
    parts.append('<linearGradient id="hypGrad" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stop-color="#eef3f8"/><stop offset="1" stop-color="#e7f0e9"/></linearGradient>')
    parts.append('</defs>')
    parts.append('<rect width="940" height="790" rx="28" fill="#ffffff"/>')
    parts.append('<rect x="24" y="24" width="892" height="742" rx="22" fill="#ffffff" stroke="#c5d9e8"/>')
    parts.append('<text x="470" y="66" text-anchor="middle" font-size="33" font-weight="900" fill="#000000">Rank structure meets hyperbolic geometry</text>')
    parts.append('<text x="470" y="97" text-anchor="middle" font-size="17" fill="#000000">LoRS supplies low-rank image-text relations.</text>')
    parts.append('<text x="470" y="121" text-anchor="middle" font-size="17" fill="#000000">MERU supplies hyperbolic image-text hierarchy.</text>')

    # LoRS source block.
    parts.append('<g transform="translate(58,132)">')
    parts.append('<rect x="0" y="0" width="350" height="320" rx="22" fill="#f7f1e8" stroke="#c5d9e8"/>')
    parts.append('<text x="26" y="44" font-size="32" font-weight="900" fill="#000000">LoRS <tspan font-size="16" font-weight="850">(ICML 2024)</tspan></text>')
    parts.append('<text x="26" y="76" font-size="16" font-weight="850" fill="#0f8f79">low-rank image-text relations</text>')
    parts.append('<g transform="translate(27,106)">')
    parts.append('<rect x="0" y="0" width="120" height="108" rx="12" fill="#ffffff" stroke="#c5d9e8"/>')
    for i in range(5):
        for j in range(6):
            fill = ["#78a99a", "#9fb7d9", "#f2dfaa", "#d49a8c"][(i + j) % 4]
            op = 0.86 if i in (1, 2, 3) or j in (1, 4) else 0.34
            parts.append(f'<rect x="{14+j*16}" y="{14+i*15}" width="11" height="11" rx="2" fill="{fill}" opacity="{op}"/>')
    parts.append('<text x="60" y="136" text-anchor="middle" font-size="16" font-weight="900" fill="#000000">S</text>')
    parts.append('<text x="148" y="60" font-size="26" font-weight="900" fill="#000000">~=</text>')
    parts.append('<rect x="184" y="8" width="46" height="92" rx="10" fill="#e7f0e9" stroke="#cbdccd"/>')
    parts.append('<rect x="236" y="31" width="74" height="46" rx="10" fill="#eef3f8" stroke="#cfdae8"/>')
    parts.append('<text x="207" y="136" text-anchor="middle" font-size="16" font-weight="900" fill="#000000">U</text>')
    parts.append('<text x="273" y="136" text-anchor="middle" font-size="16" font-weight="900" fill="#000000">V<tspan baseline-shift="super">T</tspan>R</text>')
    parts.append('</g>')
    pill(175, 266, "Rank-Aware", "#e7f0e9", "#cbdccd", color="#4f887d", size=17)
    parts.append('</g>')

    # MERU source block.
    parts.append('<g transform="translate(532,132)">')
    parts.append('<rect x="0" y="0" width="350" height="320" rx="22" fill="#f7f1e8" stroke="#c5d9e8"/>')
    parts.append('<text x="26" y="44" font-size="32" font-weight="900" fill="#000000">MERU <tspan font-size="16" font-weight="850">(ICML 2024)</tspan></text>')
    parts.append('<text x="26" y="76" font-size="16" font-weight="850" fill="#1d76bb">hierarchical image-text geometry</text>')
    parts.append('<g transform="translate(42,94)">')
    parts.append('<path d="M52 12C88 62 100 118 120 178C140 118 152 62 188 12C150 31 90 31 52 12Z" fill="url(#hypGrad)" stroke="#c5d9e8" stroke-width="2.2"/>')
    for cy, rx, ry in [(30,68,14),(64,54,11),(102,38,8),(140,24,6),(170,10,3.5)]:
        parts.append(f'<ellipse cx="120" cy="{cy}" rx="{rx}" ry="{ry}" fill="none" stroke="#d9cec1" stroke-width="1.4"/>')
    parts.append('<circle cx="120" cy="168" r="13" fill="#f2dfaa" stroke="#ffffff" stroke-width="4"/>')
    parts.append('<circle cx="83" cy="52" r="11" fill="#9fb7d9" stroke="#ffffff" stroke-width="3"/>')
    parts.append('<rect x="150" y="46" width="22" height="22" rx="7" fill="#78a99a" stroke="#ffffff" stroke-width="3"/>')
    parts.append('<path d="M120 168C110 120 100 78 83 52M120 168C132 120 143 78 161 57" fill="none" stroke="#8f8378" stroke-width="2.2"/>')
    parts.append('<text x="214" y="48" font-size="16" font-weight="850" fill="#000000">specific</text>')
    parts.append('<text x="214" y="174" font-size="16" font-weight="850" fill="#000000">generic</text>')
    parts.append('</g>')
    pill(175, 266, "Hyperbolic", "#eef3f8", "#cfdae8", color="#5d7899", size=15)
    parts.append('</g>')

    # Arrows into RAHA.
    parts.append('<path d="M260 472C322 518 372 552 416 584" fill="none" stroke="#1d76bb" stroke-width="4.8" stroke-linecap="round" marker-end="url(#arrowMot)"/>')
    parts.append('<path d="M680 472C618 518 568 552 524 584" fill="none" stroke="#1d76bb" stroke-width="4.8" stroke-linecap="round" marker-end="url(#arrowMot)"/>')

    # RAHA destination block.
    parts.append('<g transform="translate(190,602)">')
    parts.append('<rect x="0" y="0" width="560" height="150" rx="24" fill="url(#rankGrad)" stroke="#cbdccd" stroke-width="2"/>')
    parts.append('<text x="280" y="47" text-anchor="middle" font-size="40" font-weight="950" fill="#000000">RAHA</text>')
    parts.append('<text x="280" y="80" text-anchor="middle" font-size="18" font-weight="850" fill="#000000">Rank-Aware Hyperbolic Alignment</text>')
    pill(132, 102, "range relevance", "#ffffff", "#c5d9e8", size=15)
    pill(280, 102, "hyperbolic ITC", "#ffffff", "#c5d9e8", size=15)
    pill(428, 102, "residual control", "#ffffff", "#c5d9e8", size=15)
    parts.append('</g>')
    parts.append('</svg>')
    return scaled_figure_svg("".join(parts))

def placeholder_svg(label: str) -> str:
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 820 260">{SVG_TEXT_STYLE}<rect width="820" height="260" fill="#ffffff"/><text x="410" y="130" text-anchor="middle" font-size="20" fill="#000000">{esc(label)}</text></svg>'


def build() -> None:
    GEN.mkdir(parents=True, exist_ok=True)
    svgs = {
        "retrieval": retrieval_svg(),
        "transfer": transfer_svg(),
        "classification": classification_svg(),
        "semantic": semantic_svg(),
        "motivation": motivation_svg(),
    }
    for name, svg in svgs.items():
        write(GEN / f"{name}.svg", svg)
    pngs = make_png_crops()
    retrieval_table, transfer_table, cls_table = make_tables()

    replacements = {
        "%%FONT_FACE%%": font_face_css(),
        "%%RAHA_MARK%%": svg_uri(raha_mark_svg()),
        "%%KAIST_LOGO%%": data_uri(KAIST_LOGO, "image/png") if KAIST_LOGO.exists() else "",
        "%%ETRI_LOGO%%": data_uri(ETRI_LOGO, "image/png") if ETRI_LOGO.exists() else "",
        "%%RETRIEVAL_FIG%%": svg_uri(svgs["retrieval"]),
        "%%TRANSFER_FIG%%": svg_uri(svgs["transfer"]),
        "%%CLASSIFICATION_FIG%%": svg_uri(svgs["classification"]),
        "%%SEMANTIC_FIG%%": svg_uri(svgs["semantic"]),
        "%%MOTIVATION_FIG%%": svg_uri(svgs["motivation"]),
        "%%ABLATION_FIG%%": pngs["ablation"].relative_to(ROOT).as_posix() if "ablation" in pngs else svg_uri(placeholder_svg("Fig. 2 crop unavailable")),
        "%%FIG_A6%%": pngs["fig_a6"].relative_to(ROOT).as_posix() if "fig_a6" in pngs else svg_uri(placeholder_svg("Fig. A6 crop unavailable")),
        "%%MAIN_QUAL%%": data_uri(pngs["main_qual"], "image/png") if "main_qual" in pngs else svg_uri(placeholder_svg("Main qualitative crop unavailable")),
        "%%QUAL_CIFAR%%": data_uri(pngs["qual_cifar"], "image/png") if "qual_cifar" in pngs else svg_uri(placeholder_svg("CIFAR qualitative crop unavailable")),
        "%%QUAL_CUB%%": data_uri(pngs["qual_cub"], "image/png") if "qual_cub" in pngs else svg_uri(placeholder_svg("CUB qualitative crop unavailable")),
        "%%QUAL_CARS%%": data_uri(pngs["qual_cars"], "image/png") if "qual_cars" in pngs else svg_uri(placeholder_svg("Cars qualitative crop unavailable")),
        "%%QUAL_IMAGENET%%": data_uri(pngs["qual_imagenet"], "image/png") if "qual_imagenet" in pngs else svg_uri(placeholder_svg("ImageNet qualitative crop unavailable")),
        "%%RETRIEVAL_ROWS%%": retrieval_table,
        "%%TRANSFER_ROWS%%": transfer_table,
        "%%CLASSIFICATION_ROWS%%": cls_table,
    }
    html_text = HTML_TEMPLATE
    for key, value in replacements.items():
        html_text = html_text.replace(key, value)
    write(ROOT / "index.html", html_text)


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RAHA | Rank-Aware Hyperbolic Alignment</title>
  <meta name="description" content="Project page for RAHA: Rank-Aware Hyperbolic Alignment for Vision-Language Dataset Distillation, ECCV 2026.">
  <link rel="icon" type="image/svg+xml" href="%%RAHA_MARK%%">
  <style>
    %%FONT_FACE%%
    :root {
      color-scheme: light;
      --paper: #ffffff;
      --paper-2: #ffffff;
      --paper-3: #f7fbff;
      --ink: #000000;
      --muted: #000000;
      --line: #c5d9e8;
      --panel: #ffffff;
      --sage: #76a99a;
      --rose: #d49a8c;
      --sky: #a7d6f3;
      --eccv-blue: #1d76bb;
      --butter: #ead48f;
      --lavender: #c3b2dc;
      --moss: #bfd4a7;
      --max: 1180px;
      --shadow: 7px 8px 0 rgba(29,118,187,.065), 0 18px 42px rgba(29,118,187,.07);
      --font-root: Pretendard;
      font-family: var(--font-root);
    }
    * { box-sizing: border-box; }
    button, input, textarea, select { font: inherit; }
    html { scroll-behavior: smooth; }
    body {
      margin: 0;
      color: var(--ink);
      font-size: 17px;
      background: #ffffff;
      line-height: 1.55;
    }
    body::before {
      content: "";
      position: fixed;
      inset: 0;
      pointer-events: none;
      background:
        linear-gradient(90deg, rgba(29, 118, 187, .026) 1px, transparent 1px),
        linear-gradient(rgba(29, 118, 187, .022) 1px, transparent 1px);
      background-size: 44px 44px;
      mask-image: linear-gradient(to bottom, rgba(0,0,0,.34), transparent 72%);
      z-index: -1;
    }
    a { color: inherit; text-decoration: none; }
    img, svg { display: block; max-width: 100%; }
    .progress { position: fixed; top: 0; left: 0; width: 100%; height: 4px; z-index: 50; background: transparent; }
    .progress span { display: block; width: 0; height: 100%; background: linear-gradient(90deg, var(--sage), var(--sky), var(--rose)); }
    .nav {
      position: sticky; top: 0; z-index: 40;
      background: rgba(255,255,255,.9);
      backdrop-filter: blur(18px);
      border-bottom: 1px solid rgba(29,118,187,.16);
    }
    .nav-inner {
      width: min(var(--max), calc(100% - 36px));
      min-height: 66px;
      margin: 0 auto;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 18px;
    }
    .brand { display: inline-flex; align-items: center; gap: 10px; font-weight: 900; letter-spacing: 0; }
    .brand-mark { width: 34px; height: 34px; display: block; flex: 0 0 auto; }
    .nav-links { display: flex; gap: 5px; flex-wrap: wrap; justify-content: flex-end; }
    .nav-links a { padding: 7px 10px; border-radius: 999px; color: var(--muted); font-size: .94rem; font-weight: 750; }
    .nav-links a:hover, .nav-links a.active { color: var(--ink); background: #ffffff; }
    .wrap { width: min(var(--max), calc(100% - 36px)); margin: 0 auto; }
    .hero { padding: 64px 0 54px; }
    .hero-grid { display: grid; grid-template-columns: 1fr; gap: 30px; align-items: center; }
    .hero-copy { width: 100%; min-width: 0; text-align: center; }
    h1, h2, h3, p { margin-top: 0; }
    h1 { letter-spacing: 0; }
    .paper-title { width: 100%; max-width: none; min-width: 0; font-size: 4.15rem; line-height: 1.02; margin: 0 0 18px; font-weight: 900; text-align: center; }
    .paper-title .title-line { display: block; width: 100%; max-width: 100%; white-space: nowrap; overflow-wrap: normal; }
    .paper-title .title-main { color: var(--ink); font-weight: 900; }
    .paper-title .title-sub { color: #6f7882; font-size: .86em; font-weight: 760; }
    .paper-title .title-initial { text-decoration: underline; text-decoration-thickness: .08em; text-underline-offset: .09em; text-decoration-color: currentColor; }
    .venue { display: inline-flex; align-items: center; min-height: 32px; padding: 4px 12px; border-radius: 999px; background: #f2dfaa; border: 1px solid #d8c47e; font-weight: 900; margin: 0 0 10px; }
    .workshop { color: var(--muted); font-weight: 700; margin: 0 0 18px; }
    .workshop a, .authors a { color: var(--ink); font-weight: 850; }
    .authors { color: var(--ink); font-weight: 720; margin-bottom: 6px; }
    .authors sup { color: #000000; font-size: .68em; margin-left: 1px; }
    .affiliations { display: flex; justify-content: center; align-items: center; flex-wrap: wrap; gap: 16px; color: #000000; font-size: .94rem; font-weight: 650; margin-bottom: 18px; }
    .affiliations sup { font-size: .72em; margin-right: 1px; }
    .affil-item { display: inline-flex; align-items: center; gap: 6px; white-space: nowrap; }
    .affil-logo { display: inline-block; width: auto; height: 18px; max-width: 74px; object-fit: contain; flex: 0 0 auto; opacity: .9; filter: none; }
    .kaist-logo { height: 18px; max-width: 72px; }
    .etri-logo { height: 17px; max-width: 70px; }
    .tldr {
      position: relative;
      width: 100%;
      background: #ffffff;
      color: var(--ink);
      border: 1px solid #a7d6f3;
      border-left: 8px solid var(--sage);
      border-radius: 8px;
      padding: 18px 20px;
      box-shadow: var(--shadow);
      margin: 22px 0 0;
      text-align: center;
    }
    .tldr b { display: block; color: var(--ink); font-size: .9rem; letter-spacing: .1em; text-transform: uppercase; margin-bottom: 6px; text-align: left; }
    .actions { width: 100%; display: flex; justify-content: center; flex-wrap: wrap; gap: 9px; margin-top: 18px; }
    .button {
      width: auto; min-height: 34px; display: inline-flex; align-items: center; justify-content: center; gap: 6px;
      padding: 6px 11px; border: 1px solid rgba(29,118,187,.22); border-radius: 7px;
      background: rgba(255,255,255,.92); box-shadow: none; font-size: .98rem; font-weight: 850;
      transition: transform .16s ease, box-shadow .16s ease, background .16s ease;
    }
    .button:hover { transform: translateY(-1px); box-shadow: 0 8px 18px rgba(29,118,187,.10); background: #ffffff; }
    .button.primary { background: #f2dfaa; border-color: #d8c47e; }
    .button svg { width: 16px; height: 16px; stroke-width: 2.2; }
    .hero-visual {
      background: rgba(255,255,255,.92);
      border: 1px solid rgba(29,118,187,.16);
      border-radius: 14px;
      box-shadow: var(--shadow);
      padding: 10px;
      width: 100%;
    }
    .band { padding: 76px 0; border-top: 1px solid rgba(29,118,187,.12); }
    .band.dark, .citation-band { background: rgba(255,255,255,.82); color: var(--ink); }
    .section-head { max-width: 860px; margin-bottom: 32px; }
    .kicker { color: var(--ink); font-weight: 880; letter-spacing: .09em; text-transform: uppercase; font-size: .8rem; margin-bottom: 8px; }
    .dark .kicker, .citation-band .kicker { color: var(--ink); }
    h2 { font-size: 3.25rem; line-height: 1; letter-spacing: 0; margin-bottom: 14px; font-weight: 880; }
    h3 { font-size: 1.15rem; line-height: 1.2; margin-bottom: 8px; }
    .lead { color: var(--muted); font-size: 1.12rem; max-width: 880px; }
    .dark .lead, .citation-band .lead { color: var(--muted); }
    .split { display: grid; grid-template-columns: minmax(0, .96fr) minmax(320px, 1.04fr); gap: 28px; align-items: start; }
    .grid-3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; }
    .motivation-notes { display: grid; grid-template-columns: 1fr; gap: 12px; margin-top: 18px; }
    .motivation-notes .tile { box-shadow: var(--shadow); }
    .grid-2 { display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; }
    .figure-table-grid { display: grid; grid-template-columns: minmax(0, 1.18fr) minmax(360px, .82fr); gap: 16px; align-items: start; }
    .figure-table-grid.reverse { grid-template-columns: minmax(360px, .82fr) minmax(0, 1.18fr); }
    .figure-table-grid > .figure, .figure-table-grid > .table-shell { min-width: 0; }
    .figure-table-grid.reverse.transfer-layout { grid-template-columns: minmax(0, .72fr) minmax(480px, 1.28fr); }
    .transfer-layout .table-shell { overflow-x: hidden; }
    .transfer-table { table-layout: fixed; }
    .transfer-table th, .transfer-table td { padding: 9px 8px; font-size: .88rem; line-height: 1.22; white-space: normal; }
    .transfer-table th { font-size: .68rem; letter-spacing: .025em; line-height: 1.12; }
    .transfer-table th:nth-child(1), .transfer-table td:nth-child(1), .transfer-table th:nth-child(6), .transfer-table td:nth-child(6) { width: 12%; }
    .classification-layout { grid-template-columns: minmax(0, 1.1fr) minmax(360px, .9fr); }
    .figure-gap { height: 16px; }
    .tile, .figure, .table-shell {
      background: rgba(255,255,255,.96);
      border: 1px solid rgba(29,118,187,.16);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }
    .tile { padding: 18px; }
    .tile strong { display: block; font-size: 2.2rem; line-height: 1; margin-bottom: 8px; font-weight: 900; }
    .tile p { margin-bottom: 0; color: var(--muted); }
    .figure { overflow: hidden; }
    .figure img { width: 100%; height: auto; background: #ffffff; }
    .caption { padding: 11px 13px 13px; color: var(--muted); font-size: .96rem; border-top: 1px solid rgba(29,118,187,.12); background: #ffffff; }
    .dark .figure, .citation-band .figure { background: #ffffff; color: var(--ink); }
    .dark .caption, .citation-band .caption { color: var(--muted); }
    .method-strip { display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px; margin-top: 28px; }
    .step { min-height: 145px; padding: 14px; background: #ffffff; border: 1px solid rgba(29,118,187,.16); border-radius: 8px; color: var(--ink); box-shadow: 6px 7px 0 rgba(29,118,187,.055), 0 10px 24px rgba(29,118,187,.07); }
    .step span { display: inline-grid; place-items: center; width: 29px; height: 29px; border-radius: 50%; background: #f2dfaa; border: 1px solid #d8c47e; font-weight: 900; margin-bottom: 10px; }
    .equation-card { background: #ffffff; border: 1px solid var(--line); border-radius: 8px; padding: 24px 28px; margin-top: 18px; text-align: center; box-shadow: var(--shadow); }
    .equation { font-family: var(--font-root); font-size: 2.18rem; font-style: normal; font-weight: 780; line-height: 1.32; color: var(--ink); }
    .equation sub { font-size: .56em; vertical-align: -0.32em; font-weight: 680; color: #000000; }
    .equation .lambda { font-family: var(--font-root); font-style: italic; font-weight: 700; }
    .equation-sub { color: var(--muted); margin: 10px 0 0; font-size: .98rem; }
    .table-shell { overflow-x: auto; }
    table { width: 100%; border-collapse: collapse; min-width: 0; }
    th, td { padding: 10px 12px; border-bottom: 1px solid var(--line); text-align: left; font-size: .97rem; }
    th { background: #ffffff; font-size: .82rem; text-transform: uppercase; letter-spacing: .04em; color: var(--ink); }
    tr:last-child td { border-bottom: 0; }
    .dataset-flickr8k { --dataset-accent: #6f9fd0; --dataset-soft: #eaf3fb; }
    .dataset-flickr30k { --dataset-accent: #76a99a; --dataset-soft: #e9f3ef; }
    .dataset-coco { --dataset-accent: #d49a8c; --dataset-soft: #f8ebe7; }
    .pair-100 { --pair-accent: #c3b2dc; --pair-soft: #f2eef8; }
    .pair-200 { --pair-accent: #9fb7d9; --pair-soft: #edf4fb; }
    .pair-500 { --pair-accent: #ead48f; --pair-soft: #fbf6df; }
    .pair-1000 { --pair-accent: #bfd4a7; --pair-soft: #f0f6ea; }
    .dataset-pill, .pair-pill {
      display: inline-flex; align-items: center; justify-content: center; min-height: 26px;
      padding: 3px 9px; border-radius: 999px; border: 1px solid currentColor;
      color: #000000; font-weight: 850; line-height: 1;
    }
    .dataset-pill { background: var(--dataset-soft); border-color: var(--dataset-accent); }
    .pair-pill { background: var(--pair-soft); border-color: var(--pair-accent); min-width: 54px; }
    .delta {
      --dataset-accent: #c5d9e8; --dataset-soft: #f7fbff; --pair-accent: #c5d9e8; --pair-soft: #ffffff;
      color: #000000; font-weight: 900; text-align: center;
      background: #e7f0e9; box-shadow: inset 5px 0 0 #76a99a, inset 0 0 0 1px #76a99a;
    }
    .delta.neg { background: #f6e7dc; box-shadow: inset 5px 0 0 #d49a8c, inset 0 0 0 1px #d49a8c; }
    .retrieval-table .delta {
      background: linear-gradient(135deg, var(--dataset-soft), var(--pair-soft));
      box-shadow: inset 5px 0 0 var(--dataset-accent), inset -5px 0 0 var(--pair-accent), inset 0 0 0 1px rgba(29,118,187,.20);
    }
    .retrieval-table .delta.neg { background: linear-gradient(135deg, #f6e7dc, var(--pair-soft)); }
    .qual-grid, .analysis-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; align-items: start; }
    .qual-label { padding: 9px 12px; font-weight: 850; background: #ffffff; color: var(--ink); border-bottom: 1px solid #c5d9e8; }
    .qual-grid .figure, .analysis-grid .figure { display: flex; flex-direction: column; }
    .media { background: #ffffff; overflow: hidden; line-height: 0; }
    .qual-grid .media img, .analysis-grid .media img { width: 100%; height: auto; object-fit: initial; }
    .classification-chart img { aspect-ratio: auto; object-fit: initial; }
    .classification-qual-grid { grid-template-columns: 1fr; }
    code, pre { font-family: var(--font-root); }
    pre { margin: 14px 0 0; overflow-x: auto; background: #ffffff; color: var(--ink); border: 1px solid rgba(29,118,187,.16); border-radius: 8px; padding: 16px; box-shadow: var(--shadow); }
    .copy-button { border: 1px solid rgba(29,118,187,.22); color: var(--ink); background: #ffffff; border-radius: 8px; min-height: 38px; padding: 7px 12px; font-weight: 850; cursor: pointer; }
    .citation-box { display: flex; justify-content: space-between; gap: 18px; align-items: end; }
    .fade { opacity: 0; transform: translateY(18px); transition: opacity .62s ease, transform .62s ease; }
    .fade.visible { opacity: 1; transform: translateY(0); }
    @media (prefers-reduced-motion: reduce) { html { scroll-behavior: auto; } .fade { opacity: 1; transform: none; transition: none; } .button:hover { transform: none; } }
    @media (max-width: 980px) { .hero-grid, .split, .grid-2, .grid-3, .figure-table-grid, .figure-table-grid.reverse, .classification-layout { grid-template-columns: 1fr; } .method-strip { grid-template-columns: 1fr; } .qual-grid, .analysis-grid { grid-template-columns: 1fr; } .paper-title { font-size: 3rem; } h2 { font-size: 2.65rem; } }
    @media (max-width: 760px) { body { font-size: 16px; } .nav-inner { flex-direction: column; align-items: flex-start; padding: 12px 0; } .wrap, .nav-inner { width: min(var(--max), calc(100% - 24px)); } .hero { padding-top: 42px; } .hero-visual { padding: 7px; } .citation-box { align-items: start; flex-direction: column; } table { min-width: 620px; } .transfer-layout table { min-width: 0; } .transfer-table th, .transfer-table td { padding: 7px 4px; font-size: .72rem; } .transfer-table th { font-size: .56rem; letter-spacing: 0; } .paper-title { font-size: 2.35rem; } h2 { font-size: 2.15rem; } .equation { font-size: 1.55rem; } }
  </style>
  <noscript><style>.fade{opacity:1;transform:none}</style></noscript>
</head>
<body>
  <div class="progress" aria-hidden="true"><span></span></div>
  <nav class="nav" aria-label="Main navigation">
    <div class="nav-inner">
      <a class="brand" href="#top"><img class="brand-mark" src="%%RAHA_MARK%%" alt="" aria-hidden="true"> RAHA</a>
      <div class="nav-links">
        <a href="#abstract">Abstract</a>
        <a href="#motivation">Motivation</a>
        <a href="#method">Method</a>
        <a href="#results">Results</a>
        <a href="#transfer">Transfer</a>
        <a href="#classification">Classification</a>
        <a href="#citation">Citation</a>
      </div>
    </div>
  </nav>

  <main id="top">
    <section class="hero">
      <div class="wrap hero-grid">
        <div class="hero-copy fade">
          <h1 class="paper-title"><span class="title-line title-main"><span class="title-initial">R</span>ank-<span class="title-initial">A</span>ware <span class="title-initial">H</span>yperbolic <span class="title-initial">A</span>lignment</span><span class="title-line title-sub">for Vision-Language Dataset Distillation</span></h1>
          <p class="venue">ECCV 2026</p>
          <!-- <p class="workshop">Also presented at <span>CDEL Workshop</span></p> -->
          <p class="authors"><a href="https://andyj1.github.io/" target="_blank" rel="noreferrer">Jongoh Jeong</a><sup>1</sup>, Sun-Kyung Lee<sup>2</sup>, Kuk-Jin Yoon<sup>1</sup></p>
          <p class="affiliations">
            <span class="affil-item"><sup>1</sup><span>KAIST</span><img class="affil-logo kaist-logo" src="%%KAIST_LOGO%%" alt="" aria-hidden="true"></span>
            <span class="affil-item"><sup>2</sup><span>ETRI</span><img class="affil-logo etri-logo" src="%%ETRI_LOGO%%" alt="" aria-hidden="true"></span>
          </p>
          <div class="actions">
            <a class="button primary" href="https://andyj1.github.io/raha/docs/paper.pdf" target="_blank" rel="noreferrer"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M7 3h7l5 5v13H7z"></path><path d="M14 3v5h5"></path></svg>Paper</a>
            <a class="button" href="https://andyj1.github.io/raha/docs/slides.pdf" target="_blank" rel="noreferrer"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><rect x="3" y="4" width="18" height="12" rx="1"></rect><path d="M12 16v4M8 20h8"></path></svg>Slides</a>
            <a class="button" href="https://andyj1.github.io/raha/docs/poster.pdf" target="_blank" rel="noreferrer"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><rect x="5" y="3" width="14" height="18" rx="1"></rect><path d="M8 7h8M8 11h8M8 15h5"></path></svg>Poster</a>
            <a class="button" href="https://andyj1.github.io/raha" target="_blank" rel="noreferrer"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="m16 18 6-6-6-6"></path><path d="m8 6-6 6 6 6"></path></svg>Code</a>
            <a class="button" href="#citation"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M8 7h8M8 12h8M8 17h5"></path><path d="M5 3h14v18H5z"></path></svg>BibTeX</a>
          </div>
          <div class="tldr"><b>TL;DR</b>RAHA distills compact image-text datasets by combining rank-aware semantic range matching with hyperbolic image-text alignment, preserving dominant shared relations while compressing weak residual correlations that would otherwise waste synthetic capacity.</div>
        </div>
        <div class="hero-visual fade"><img src="%%SEMANTIC_FIG%%" alt="Diagram comparing Euclidean hyperspherical image-text embeddings with Lorentz hyperboloid image-text embeddings."></div>
      </div>
    </section>

    <section class="band" id="abstract">
      <div class="wrap split">
        <div class="section-head fade">
          <div class="kicker">Abstract</div>
          <h2>Compress paired data without flattening its hierarchy.</h2>
          <p class="lead">Vision-language dataset distillation compresses large image-text datasets into compact synthetic sets so vision-language models can be trained efficiently while retaining training utility and downstream performance. RAHA adds a geometry-aware objective: lift representations to hyperbolic space, expose the rank-k shared semantic range, then regulate the residual subspace instead of aligning every direction uniformly.</p>
        </div>
        <div class="grid-3 fade">
          <article class="tile"><strong>+9.6 pp</strong><p>Flickr30k mean retrieval at 1000 pairs over Baseline: 38.0 vs. 28.4.</p></article>
          <article class="tile"><strong>+6.5 pp</strong><p>Flickr8k mean retrieval at 1000 pairs over Baseline. The 200-pair gain is already +5.8 pp.</p></article>
          <article class="tile"><strong>+4.4 pp</strong><p>COCO mean retrieval at 1000 pairs over Baseline, with +4.0 pp Flickr8k transfer at 500 pairs.</p></article>
        </div>
      </div>
    </section>

    <section class="band" id="motivation">
      <div class="wrap split">
        <div class="section-head fade">
          <div class="kicker">Motivation</div>
          <h2>Rank structure meets hyperbolic hierarchy.</h2>
          <p class="lead">RAHA is motivated by two complementary observations. LoRS shows that image-text relations can be explicitly modeled through a low-rank similarity structure, which motivates the rank-aware range decomposition. MERU shows why hyperbolic geometry is useful for cross-modal abstraction: a generic caption can describe many possible images, while a specific image-text instance occupies a more detailed semantic region.</p>
          <div class="motivation-notes">
            <article class="tile"><h3>From LoRS</h3><p>Preserve the dominant low-rank image-text relevance pattern instead of treating every pairwise direction uniformly.</p></article>
            <article class="tile"><h3>From MERU</h3><p>Keep generic captions close to many descendants while leaving specific image-text instances separable.</p></article>
          </div>
        </div>
        <div class="figure fade"><img src="%%MOTIVATION_FIG%%" alt="Motivation diagram showing LoRS and MERU feeding into RAHA."><div class="caption">High-level motivation: LoRS contributes the rank-aware relation view, and MERU contributes the hyperbolic image-text geometry view.</div></div>
      </div>
    </section>

    <section class="band dark" id="method">
      <div class="wrap">
        <div class="section-head fade">
          <div class="kicker">Method</div>
          <h2>Lift. Split. Match relevance.</h2>
          <p class="lead">The pipeline stays compact: synthetic pairs are optimized through hyperbolic contrastive alignment plus range/residual relevance distillation.</p>
        </div>
        <div class="method-strip fade">
          <div class="step"><span>1</span><h3>Synthetic pairs</h3><p>Learn pixels and continuous text embeddings.</p></div>
          <div class="step"><span>2</span><h3>Lorentz lift</h3><p>Use geodesic image-text contrastive alignment.</p></div>
          <div class="step"><span>3</span><h3>Tangent SVD</h3><p>Estimate the real batch coupling rank.</p></div>
          <div class="step"><span>4</span><h3>Range match</h3><p>Preserve dominant shared semantic order.</p></div>
          <div class="step"><span>5</span><h3>Residual control</h3><p>Keep weak interactions from dominating.</p></div>
        </div>
        <div class="equation-card fade">
          <div class="equation">L<sub>total</sub> = L<sub>hITC</sub> + <span class="lambda">&lambda;</span><sub>range</sub>L<sub>range</sub> + <span class="lambda">&lambda;</span><sub>residual</sub>L<sub>residual</sub></div>
        </div>
      </div>
    </section>

    <section class="band" id="analysis">
      <div class="wrap">
        <div class="section-head fade">
          <div class="kicker">Analysis</div>
          <h2>Range matching carries the retrieval signal, and residual compression keeps it usable.</h2>
          <p class="lead">Fig. 2 and Fig. A6 give the component-level view: RAHA is strongest when hyperbolic contrastive alignment is paired with dominant range supervision, controlled residual refinement, and hyperbolic rather than Euclidean relevance matching.</p>
        </div>
        <div class="analysis-grid">
          <div class="figure fade"><div class="media"><img src="%%ABLATION_FIG%%" alt="Fig. 2 ablation study with each RAHA component added."></div><div class="caption">Fig. 2: Component attribution under the same protocol. Range matching provides the main relevance-transfer gain, while residual control improves the full model as a refinement when anchored by the range.</div></div>
          <div class="figure fade"><div class="media"><img src="%%FIG_A6%%" alt="Fig. A6 hyperparameter sensitivity and geometry ablation of RAHA."></div><div class="caption">Fig. A6: Hyperparameter and geometry ablations. The same relevance-matching pipeline degrades in the all-Euclidean variant, while hyperbolic range/residual matching recovers stable retrieval behavior.</div></div>
        </div>
      </div>
    </section>

    <section class="band" id="results">
      <div class="wrap">
        <div class="section-head fade">
          <div class="kicker">Retrieval</div>
          <h2>Retrieval across synthetic budgets.</h2>
          <p class="lead">The chart and table show mean retrieval for 100, 200, 500, and 1000 pairs. The paper's asterisked batch-size variant is folded into Baseline, and each dataset panel uses the nearest y-axis ceiling above the strongest method.</p>
        </div>
        <div class="figure-table-grid retrieval-layout">
          <div class="figure fade"><img src="%%RETRIEVAL_FIG%%" alt="Retrieval mean recall across 100, 200, 500, and 1000 synthetic pairs."><div class="caption">Mean recall across Flickr8k, Flickr30k, and COCO. Axes are tightened per dataset. Higher is better.</div></div>
          <div class="table-shell fade"><table class="retrieval-table"><thead><tr><th>Dataset</th><th>Pairs</th><th>Baseline mean</th><th>RAHA mean</th><th>Δ</th></tr></thead><tbody>%%RETRIEVAL_ROWS%%</tbody></table></div>
        </div>
      </div>
    </section>

    <section class="band" id="transfer">
      <div class="wrap">
        <div class="section-head fade">
          <div class="kicker">Transfer</div>
          <h2>Distilled relevance survives encoder architecture change.</h2>
          <p class="lead">Cross-architecture transfer is reported on Flickr8k by replacing image backbones and text encoders while keeping the distilled set fixed. RAHA separates most clearly at 200 and 500 pairs.</p>
        </div>
        <div class="figure-table-grid reverse transfer-layout">
          <div class="table-shell fade"><table class="transfer-table" aria-label="Cross-architecture transfer results"><thead><tr><th>Pairs</th><th>Baseline<br>source</th><th>RAHA<br>source</th><th>Baseline<br>transfer</th><th>RAHA<br>transfer</th><th>Δ</th></tr></thead><tbody>%%TRANSFER_ROWS%%</tbody></table></div>
          <div class="figure fade"><img src="%%TRANSFER_FIG%%" alt="Cross-architecture transfer chart for Baseline and RAHA at 100, 200, and 500 pairs."><div class="caption">Cross-architecture transfer mean. The distilled set is a reusable offline artifact. Higher is better.</div></div>
        </div>
      </div>
    </section>

    <section class="band" id="qualitative">
      <div class="wrap">
        <div class="section-head fade">
          <div class="kicker">Synthesized pairs</div>
          <h2>Less residual debris, more readable pairs.</h2>
          <p class="lead">The paper's qualitative comparison shows initialization, Baseline, and RAHA outputs. RAHA keeps stronger image-text agreement while reducing high-frequency artifacts.</p>
        </div>
        <div class="figure fade"><img src="%%MAIN_QUAL%%" alt="Fig. 1 qualitative synthesized image-text pairs comparing initialization, Baseline, and RAHA."><div class="caption">Fig. 1: qualitative synthesized image-text pairs comparing initialization, Baseline, and RAHA.</div></div>
      </div>
    </section>

    <section class="band" id="classification">
      <div class="wrap">
        <div class="section-head fade">
          <div class="kicker">Prompted classification</div>
          <h2>Prompted classification with corresponding qualitative samples.</h2>
          <p class="lead">At 100 pairs, RAHA improves top-1 prompted classification on all four benchmarks reported in the appendix. The qualitative crops below correspond to those datasets.</p>
        </div>
        <div class="figure-table-grid classification-layout">
          <div class="figure classification-chart fade"><img src="%%CLASSIFICATION_FIG%%" alt="Prompted image classification top-1 accuracy chart."><div class="caption">Top-1 accuracy at 100 pairs. Higher is better.</div></div>
          <div class="table-shell fade"><table><thead><tr><th>Dataset</th><th>Baseline</th><th>RAHA</th><th>Δ</th></tr></thead><tbody>%%CLASSIFICATION_ROWS%%</tbody></table></div>
        </div>
        <div class="figure-gap"></div>
        <div class="qual-grid classification-qual-grid">
          <div class="figure fade"><div class="qual-label">CIFAR-100</div><div class="media"><img src="%%QUAL_CIFAR%%" alt="CIFAR-100 qualitative distilled samples from the appendix."></div><div class="caption">Fig. A1: Distilled data comparison on CIFAR-100. Left: initial samples. Middle: Baseline-distilled. Right: RAHA-distilled. RAHA preserves visual fidelity and image-text consistency more reliably across samples.</div></div>
          <div class="figure fade"><div class="qual-label">CUB-200-2011</div><div class="media"><img src="%%QUAL_CUB%%" alt="CUB qualitative distilled samples from the appendix."></div><div class="caption">Fig. A2: Distilled data comparison on CUB-200-2011. Left: initial samples. Middle: Baseline-distilled. Right: RAHA-distilled.</div></div>
          <div class="figure fade"><div class="qual-label">Stanford Cars</div><div class="media"><img src="%%QUAL_CARS%%" alt="Stanford Cars qualitative distilled samples from the appendix."></div><div class="caption">Fig. A3: Distilled data comparison on Stanford Cars. Left: initial samples. Middle: Baseline-distilled. Right: RAHA-distilled.</div></div>
          <div class="figure fade"><div class="qual-label">ImageNet-1K</div><div class="media"><img src="%%QUAL_IMAGENET%%" alt="ImageNet qualitative distilled samples from the appendix."></div><div class="caption">Fig. A4: Distilled data comparison on ImageNet-1K. Left: initial samples. Middle: Baseline-distilled. Right: RAHA-distilled.</div></div>
        </div>
      </div>
    </section>

    <section class="band citation-band" id="citation">
      <div class="wrap">
        <div class="citation-box fade">
          <div><h2>Citation</h2></div>
          <button class="copy-button" type="button" data-copy="#citation-code">Copy</button>
        </div>
        <pre class="fade"><code id="citation-code">@inproceedings{jeong2026raha,
  title     = {Rank-Aware Hyperbolic Alignment for Vision-Language Dataset Distillation},
  author    = {Jeong, Jongoh and Lee, Sun-Kyung and Yoon, Kuk-Jin},
  booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
  year      = {2026}
}</code></pre>
      </div>
    </section>
  </main>


  <script>
    (function () {
      var progress = document.querySelector(".progress span");
      var fades = Array.prototype.slice.call(document.querySelectorAll(".fade"));
      var navLinks = Array.prototype.slice.call(document.querySelectorAll(".nav-links a"));
      var sections = navLinks.map(function (link) { return document.querySelector(link.getAttribute("href")); }).filter(Boolean);
      var title = document.querySelector(".paper-title");
      function fitTitle() {
        if (!title) return;
        var width = title.clientWidth;
        if (!width) return;
        title.querySelectorAll(".title-line").forEach(function (line) {
          line.style.fontSize = "";
          var base = parseFloat(window.getComputedStyle(line).fontSize);
          if (!base) return;
          var needed = line.scrollWidth;
          if (needed > width) {
            line.style.fontSize = Math.max(11.5, base * width / needed) + "px";
          }
        });
      }
      function updateProgress() {
        var doc = document.documentElement;
        var max = doc.scrollHeight - window.innerHeight;
        var pct = max > 0 ? window.scrollY / max * 100 : 0;
        progress.style.width = Math.max(0, Math.min(100, pct)) + "%";
        var probe = window.scrollY + Math.max(160, window.innerHeight * 0.35);
        var current = sections.length ? "#" + sections[0].id : "";
        sections.forEach(function (section) {
          var top = section.getBoundingClientRect().top + window.scrollY;
          if (probe >= top) current = "#" + section.id;
        });
        if (sections.length && window.scrollY + window.innerHeight >= doc.scrollHeight - 2) {
          current = "#" + sections[sections.length - 1].id;
        }
        navLinks.forEach(function (link) { link.classList.toggle("active", link.getAttribute("href") === current); });
      }
      if ("IntersectionObserver" in window) {
        var observer = new IntersectionObserver(function (entries) {
          entries.forEach(function (entry) {
            if (entry.isIntersecting) {
              entry.target.classList.add("visible");
              observer.unobserve(entry.target);
            }
          });
        }, { rootMargin: "0px 0px -10% 0px", threshold: .12 });
        fades.forEach(function (el) { observer.observe(el); });
      } else {
        fades.forEach(function (el) { el.classList.add("visible"); });
      }
      document.querySelectorAll("[data-copy]").forEach(function (button) {
        button.addEventListener("click", function () {
          var target = document.querySelector(button.dataset.copy);
          if (!target) return;
          navigator.clipboard.writeText(target.textContent.trim()).then(function () {
            var original = button.textContent;
            button.textContent = "Copied";
            setTimeout(function () { button.textContent = original; }, 1300);
          });
        });
      });
      window.addEventListener("scroll", updateProgress, { passive: true });
      window.addEventListener("resize", function () { fitTitle(); updateProgress(); });
      window.addEventListener("load", fitTitle);
      if (document.fonts && document.fonts.ready) document.fonts.ready.then(fitTitle);
      fitTitle();
      updateProgress();
    })();
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    build()
