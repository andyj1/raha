#!/usr/bin/env python3
"""Build the hero semantic transition figure for the RAHA project page."""
from __future__ import annotations

import base64
import html
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "generated_figures" / "semantic.svg"
INDEX = ROOT / "index.html"
MATCHA_ICON = ROOT / "misc" / "assets" / "matcha_leaf.svg"

WIDTH, HEIGHT = 1180, 712
PANEL_TOP = 112

COLORS = {
    "image": "#78a99a",
    "text": "#9fb7d9",
    "generic": "#f2dfaa",
    "range": "#76a99a",
    "residual": "#d49a8c",
    "ink": "#000000",
    "line": "#c5d9e8",
    "pair": "#8f8378",
}
SVG_TEXT_STYLE = '<style>text{font-family:Arial, Helvetica, sans-serif;letter-spacing:0}</style>'


def esc(text: object) -> str:
    return html.escape(str(text), quote=True)


def svg_inner(path: Path) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8")
    svg_start = text.find("<svg")
    body_start = text.find(">", svg_start)
    body_end = text.rfind("</svg>")
    if svg_start < 0 or body_start < 0 or body_end < 0:
        return ""
    return text[body_start + 1 : body_end].strip()


def node(parts: list[str], x: float, y: float, kind: str, fill: str, size: int = 13) -> None:
    if kind == "I":
        parts.append(
            f'<rect x="{x - 13:.1f}" y="{y - 13:.1f}" width="26" height="26" rx="7" '
            f'fill="{fill}" stroke="#ffffff" stroke-width="3.5"/>'
        )
    else:
        parts.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="14" fill="{fill}" '
            f'stroke="#ffffff" stroke-width="3.5"/>'
        )
    parts.append(
        f'<text x="{x:.1f}" y="{y + 4:.1f}" text-anchor="middle" font-size="{size}" '
        f'font-weight="900" fill="{COLORS["ink"]}">{kind}</text>'
    )


def pair_link(parts: list[str], ix: float, iy: float, tx: float, ty: float, opacity: float = 0.72) -> None:
    parts.append(
        f'<path d="M{ix:.1f} {iy:.1f}L{tx:.1f} {ty:.1f}" fill="none" stroke="{COLORS["pair"]}" '
        f'stroke-width="1.8" stroke-linecap="round" opacity="{opacity:.2f}"/>'
    )


def covariance_decomposition(
    parts: list[str],
    ix: float,
    iy: float,
    tx: float,
    ty: float,
    *,
    box_x: float,
    box_y: float,
    box_w: float,
    title: str,
) -> None:
    rx = ix + (tx - ix) * 0.62
    ry = iy + (ty - iy) * 0.62
    parts.append(
        f'<path d="M{ix:.1f} {iy:.1f}L{rx:.1f} {ry:.1f}" fill="none" stroke="{COLORS["range"]}" '
        f'stroke-width="4.4" stroke-linecap="round"/>'
    )
    parts.append(
        f'<path d="M{rx:.1f} {ry:.1f}L{tx:.1f} {ty:.1f}" fill="none" stroke="{COLORS["residual"]}" '
        f'stroke-width="2.8" stroke-linecap="round" stroke-dasharray="6 5"/>'
    )
    box_h = 48
    cx = box_x + box_w / 2
    parts.append(
        f'<rect x="{box_x:.1f}" y="{box_y:.1f}" width="{box_w:.1f}" height="{box_h}" rx="12" '
        f'fill="#ffffff" stroke="{COLORS["line"]}" opacity=".98"/>'
    )
    parts.append(
        f'<text x="{cx:.1f}" y="{box_y + 15:.1f}" text-anchor="middle" font-size="10.5" '
        f'font-weight="900" fill="{COLORS["ink"]}">{esc(title)}</text>'
    )
    parts.append(
        f'<line x1="{box_x + 14:.1f}" y1="{box_y + 27:.1f}" x2="{box_x + 34:.1f}" '
        f'y2="{box_y + 27:.1f}" stroke="{COLORS["range"]}" stroke-width="3.8" stroke-linecap="round"/>'
    )
    parts.append(
        f'<text x="{box_x + 40:.1f}" y="{box_y + 30:.1f}" font-size="9.8" font-weight="850" '
        f'fill="{COLORS["ink"]}">dominant range</text>'
    )
    parts.append(
        f'<line x1="{box_x + 14:.1f}" y1="{box_y + 39:.1f}" x2="{box_x + 34:.1f}" '
        f'y2="{box_y + 39:.1f}" stroke="{COLORS["residual"]}" stroke-width="2.8" stroke-linecap="round" '
        f'stroke-dasharray="5 4"/>'
    )
    parts.append(
        f'<text x="{box_x + 40:.1f}" y="{box_y + 42:.1f}" font-size="9.8" font-weight="850" '
        f'fill="{COLORS["ink"]}">residual</text>'
    )


def pill(
    parts: list[str],
    x: float,
    y: float,
    text: str,
    fill: str,
    stroke: str,
    *,
    size: int = 10,
    height: int = 20,
) -> None:
    width = max(66, len(text) * (size * 0.56) + 20)
    parts.append(
        f'<rect x="{x - width / 2:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height}" rx="10" '
        f'fill="{fill}" stroke="{stroke}"/>'
    )
    parts.append(
        f'<text x="{x:.1f}" y="{y + 14:.1f}" text-anchor="middle" font-size="{size}" '
        f'font-weight="850" fill="{COLORS["ink"]}">{esc(text)}</text>'
    )


def text_width(text: str, size: float, *, weight: str = "400") -> float:
    scale = {"400": 0.46, "850": 0.48, "900": 0.5}.get(weight, 0.47)
    return len(text) * size * scale


def wrap_lines(text: str, max_width: float, size: float, *, weight: str = "400") -> list[str]:
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        trial = " ".join([*current, word])
        if current and text_width(trial, size, weight=weight) > max_width:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return lines


def tight_multiline_box(
    parts: list[str],
    lines: list[str],
    *,
    cx: float,
    y: float,
    size: float,
    weight: str,
    fill: str,
    stroke: str,
    pad_x: float = 8,
    pad_y: float = 5,
    line_gap: float = 12,
    rx: float = 10,
) -> None:
    box_w = max(text_width(line, size, weight=weight) for line in lines) + pad_x * 2
    box_h = pad_y * 2 + line_gap * (len(lines) - 1) + size * 1.05
    x = cx - box_w / 2
    parts.append(
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{box_w:.1f}" height="{box_h:.1f}" rx="{rx}" '
        f'fill="{fill}" stroke="{stroke}"/>'
    )
    for i, line in enumerate(lines):
        parts.append(
            f'<text x="{cx:.1f}" y="{y + pad_y + size + i * line_gap:.1f}" text-anchor="middle" '
            f'font-size="{size}" font-weight="{weight}" fill="{COLORS["ink"]}">{esc(line)}</text>'
        )


def panel_footer(parts: list[str], x: float, y: float, w: float, line1: str, line2: str, fill: str, stroke: str) -> None:
    parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="34" rx="12" fill="{fill}" stroke="{stroke}"/>')
    parts.append(
        f'<text x="{x + w / 2:.1f}" y="{y + 14:.1f}" text-anchor="middle" font-size="10.5" '
        f'font-weight="900" fill="{COLORS["ink"]}">{esc(line1)}</text>'
    )
    parts.append(
        f'<text x="{x + w / 2:.1f}" y="{y + 27:.1f}" text-anchor="middle" font-size="9.8" '
        f'fill="{COLORS["ink"]}">{esc(line2)}</text>'
    )


def euclidean_panel(parts: list[str]) -> None:
    parts.append(f'<g transform="translate(36,{PANEL_TOP})">')
    parts.append('<rect x="0" y="0" width="438" height="468" rx="18" fill="#f7f1e8" stroke="#c5d9e8"/>')
    parts.append(
        '<text x="219" y="34" text-anchor="middle" font-size="17" font-weight="900" '
        f'fill="{COLORS["ink"]}">Euclidean Hypersphere</text>'
    )
    parts.append(
        '<text x="219" y="52" text-anchor="middle" font-size="10.5" fill="#000000">'
        "same image-caption pairs; flat surface mixes generic and fine directions</text>"
    )
    parts.append('<text x="54" y="78" font-size="10" font-weight="900" fill="#78a99a">image</text>')
    parts.append('<text x="372" y="78" font-size="10" font-weight="900" fill="#9fb7d9">text</text>')

    parts.append('<circle cx="219" cy="244" r="104" fill="url(#sphereFill)" stroke="#c5d9e8" stroke-width="1.8"/>')
    for ry in (30, 52, 74):
        parts.append(
            f'<ellipse cx="219" cy="244" rx="104" ry="{ry}" fill="none" stroke="#e1d6ca" stroke-width="1.4"/>'
        )

    # Fine-grained pair — same content as the hyperbolic panel.
    fine_ix, fine_iy = 108, 196
    fine_tx, fine_ty = 330, 176
    pair_link(parts, fine_ix, fine_iy, fine_tx, fine_ty, opacity=0.78)
    dog_thumbnail(parts, fine_ix, fine_iy, size=34, tint="#78a99a")
    caption_tile(parts, fine_tx, fine_ty, ["a yellow Labrador", "retriever in a park"])
    parts.append(
        f'<text x="{fine_ix:.1f}" y="{fine_iy + 28:.1f}" text-anchor="middle" font-size="9" '
        f'font-weight="900" fill="#78a99a">fine image</text>'
    )
    parts.append(
        '<rect x="56" y="148" width="72" height="20" rx="10" fill="#eef3f8" stroke="#cfdae8"/>'
    )
    parts.append(
        '<text x="92" y="162" text-anchor="middle" font-size="9" font-weight="900" fill="#000000">'
        "fine-grained</text>"
    )

    # Generic pair — same caption and image as the hyperbolic nadir pair.
    gen_ix, gen_iy = 132, 292
    gen_tx, gen_ty = 306, 292
    pair_link(parts, gen_ix, gen_iy, gen_tx, gen_ty, opacity=0.58)
    dog_thumbnail(parts, gen_ix, gen_iy, size=30, tint="#f2dfaa")
    caption_tile(parts, gen_tx, gen_ty, ["a dog"], fill="#f6f0df", stroke="#e6cfc1")
    parts.append(
        f'<text x="{gen_ix:.1f}" y="{gen_iy + 26:.1f}" text-anchor="middle" font-size="9" '
        f'font-weight="900" fill="#c9a84a">generic image</text>'
    )

    parts.append(
        '<path d="M48 196V292" fill="none" stroke="#1d76bb" stroke-width="1.4" '
        'stroke-linecap="round" marker-start="url(#softArrow)" marker-end="url(#softArrow)"/>'
    )
    parts.append(
        '<text x="34" y="248" text-anchor="middle" font-size="9" font-weight="850" fill="#1d76bb" '
        'transform="rotate(-90 34 248)">angular mix</text>'
    )

    panel_footer(
        parts,
        28,
        418,
        382,
        "Fine and generic pairs share one flat surface",
        "same yellow-Labrador and dog captions as the Lorentz panel",
        "#f6e7dc",
        "#e6cfc1",
    )
    parts.append("</g>")


def matcha_bridge(parts: list[str]) -> None:
    matcha_icon = svg_inner(MATCHA_ICON)
    parts.append(f'<g transform="translate(506,{PANEL_TOP + 58})">')
    parts.append(
        '<path d="M-26 118H-6" stroke="#1d76bb" stroke-width="2.8" stroke-linecap="round" '
        'marker-end="url(#softArrow)"/>'
    )
    parts.append(
        '<path d="M150 118H166" stroke="#1d76bb" stroke-width="2.8" stroke-linecap="round" '
        'marker-end="url(#softArrow)"/>'
    )
    parts.append(
        '<rect x="0" y="34" width="144" height="158" rx="18" fill="#ffffff" stroke="#a7d6f3" opacity=".94"/>'
    )
    parts.append(
        '<text x="72" y="58" text-anchor="middle" font-size="10" fill="#000000">'
        '<tspan font-weight="950">Match</tspan> <tspan font-weight="950">H</tspan>yperbolically</text>'
    )
    parts.append(
        '<text x="72" y="72" text-anchor="middle" font-size="10" fill="#000000">'
        '<tspan font-weight="950">A</tspan>ligned Distributions</text>'
    )
    parts.append('<circle cx="72" cy="108" r="27" fill="#ffffff" stroke="#cbdccd"/>')
    if matcha_icon:
        parts.append(f'<g transform="translate(45,84) scale(0.105)">{matcha_icon}</g>')
    else:
        parts.append(
            '<path d="M44 138C56 112 78 112 100 128C80 166 58 164 44 138Z" fill="#8cd36f" '
            'stroke="#4caf28" stroke-width="2"/>'
        )
    parts.append(
        '<text x="72" y="142" text-anchor="middle" font-size="10" font-weight="950" '
        f'fill="{COLORS["ink"]}">MATCHA</text>'
    )
    pill(parts, 72, 152, "Lorentz model", "#eef3f8", "#cfdae8", size=8.8, height=18)
    parts.append("</g>")


def dog_thumbnail(parts: list[str], cx: float, cy: float, *, size: float = 34, tint: str = "#78a99a") -> None:
    """Small illustrative dog image tile on the image flank."""
    half = size / 2
    parts.append(
        f'<rect x="{cx - half:.1f}" y="{cy - half:.1f}" width="{size:.1f}" height="{size:.1f}" rx="8" '
        f'fill="{tint}" stroke="#ffffff" stroke-width="3"/>'
    )
    parts.append(
        f'<ellipse cx="{cx - 4:.1f}" cy="{cy - 1:.1f}" rx="5.5" ry="4.5" fill="#ffffff" opacity=".92"/>'
    )
    parts.append(
        f'<circle cx="{cx - 6:.1f}" cy="{cy - 2:.1f}" r="1.2" fill="#4a5f56"/>'
    )
    parts.append(
        f'<path d="M{cx + 2:.1f} {cy - 5:.1f}c4 0 8 3 8 7c0 5-4 9-8 9s-8-4-8-9c0-4 4-7 8-7z" '
        f'fill="#ffffff" opacity=".9"/>'
    )
    parts.append(
        f'<path d="M{cx + 10:.1f} {cy - 1:.1f}l3 2l-3 2" fill="none" stroke="#4a5f56" '
        f'stroke-width="1.2" stroke-linecap="round" stroke-linejoin="round"/>'
    )


def caption_tile(
    parts: list[str],
    cx: float,
    cy: float,
    lines: list[str],
    *,
    fill: str = "#eef3f8",
    stroke: str = "#cfdae8",
    size: float = 9.2,
) -> None:
    """Text caption badge on the text flank."""
    line_gap = 11.5
    pad_x, pad_y = 9, 6
    box_w = max(text_width(line, size, weight="850") for line in lines) + pad_x * 2
    box_h = pad_y * 2 + line_gap * (len(lines) - 1) + size * 1.05
    x = cx - box_w / 2
    y = cy - box_h / 2
    parts.append(
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{box_w:.1f}" height="{box_h:.1f}" rx="10" '
        f'fill="{fill}" stroke="{stroke}"/>'
    )
    for i, line in enumerate(lines):
        parts.append(
            f'<text x="{cx:.1f}" y="{y + pad_y + size + i * line_gap:.1f}" text-anchor="middle" '
            f'font-size="{size}" font-weight="850" fill="{COLORS["ink"]}">{esc(line)}</text>'
        )


def hyperbolic_pair_arc(
    parts: list[str],
    ix: float,
    iy: float,
    tx: float,
    ty: float,
    *,
    opacity: float = 0.72,
    stroke: str = "#8f8378",
) -> None:
    mid_x = (ix + tx) / 2
    lift = min(iy, ty) - 28
    parts.append(
        f'<path d="M{ix:.1f} {iy:.1f}Q{mid_x:.1f} {lift:.1f} {tx:.1f} {ty:.1f}" fill="none" '
        f'stroke="{stroke}" stroke-width="2" stroke-linecap="round" opacity="{opacity:.2f}"/>'
    )


def hyperbolic_panel(parts: list[str]) -> None:
    parts.append(f'<g transform="translate(674,{PANEL_TOP})">')
    parts.append('<rect x="0" y="0" width="470" height="468" rx="18" fill="#f7f1e8" stroke="#c5d9e8"/>')
    parts.append(
        '<text x="235" y="34" text-anchor="middle" font-size="17" font-weight="900" '
        f'fill="{COLORS["ink"]}">Lorentz Hyperboloid</text>'
    )
    parts.append(
        '<text x="235" y="52" text-anchor="middle" font-size="10.5" fill="#000000">'
        "lift projected features; hierarchy separates fine vs. generic captions</text>"
    )

    # Hyperboloid surface and latitude rings.
    parts.append(
        '<path d="M136 104C158 140 176 182 192 226C205 264 216 300 235 346C254 300 265 264 278 226C294 182 312 140 334 104" '
        'fill="url(#hyperFill)" stroke="#c5d9e8" stroke-width="1.8" opacity=".84"/>'
    )
    for cy, rx, ry in ((116, 92, 20), (156, 74, 16), (196, 56, 12), (236, 38, 9), (282, 22, 6), (334, 9, 3.5)):
        parts.append(
            f'<ellipse cx="235" cy="{cy}" rx="{rx}" ry="{ry}" fill="none" stroke="#d9cec1" stroke-width="1.4"/>'
        )
    parts.append(
        '<path d="M235 346C222 298 210 250 196 204C186 172 174 146 158 122" fill="none" '
        'stroke="#e5dad0" stroke-width="1.6"/>'
    )
    parts.append(
        '<path d="M235 346C248 298 262 250 276 204C286 172 298 146 314 122" fill="none" '
        'stroke="#e5dad0" stroke-width="1.6"/>'
    )

    # Time-like axis and nadir marker.
    parts.append('<path d="M235 96V362" stroke="#c8baad" stroke-width="1.8" stroke-dasharray="5 6"/>')
    parts.append('<circle cx="235" cy="346" r="5" fill="#1d76bb" opacity=".55"/>')
    parts.append(
        '<text x="248" y="350" font-size="10" font-weight="900" fill="#1d76bb">nadir (generic)</text>'
    )
    parts.append(
        '<text x="248" y="118" font-size="10" font-weight="800" fill="#3f4d57">specific ↑</text>'
    )

    parts.append('<text x="118" y="78" font-size="10" font-weight="900" fill="#78a99a">image flank</text>')
    parts.append('<text x="332" y="78" font-size="10" font-weight="900" fill="#9fb7d9">text flank</text>')

    # Fine-grained pair: image on left flank, caption on right flank (higher on hyperboloid).
    fine_ix, fine_iy = 154, 188
    fine_tx, fine_ty = 316, 168
    hyperbolic_pair_arc(parts, fine_ix, fine_iy, fine_tx, fine_ty, opacity=0.82)
    dog_thumbnail(parts, fine_ix, fine_iy, size=36, tint="#78a99a")
    caption_tile(parts, fine_tx, fine_ty, ["a yellow Labrador", "retriever in a park"])
    parts.append(
        f'<text x="{fine_ix:.1f}" y="{fine_iy + 30:.1f}" text-anchor="middle" font-size="9" '
        f'font-weight="900" fill="#78a99a">fine image</text>'
    )
    parts.append(
        '<rect x="56" y="148" width="72" height="20" rx="10" fill="#eef3f8" stroke="#cfdae8"/>'
    )
    parts.append(
        '<text x="92" y="162" text-anchor="middle" font-size="9" font-weight="900" fill="#000000">'
        "fine-grained</text>"
    )

    # Generic pair at nadir: separated from fine pair along time-like axis.
    gen_ix, gen_iy = 178, 328
    gen_tx, gen_ty = 292, 328
    hyperbolic_pair_arc(parts, gen_ix, gen_iy, gen_tx, gen_ty, opacity=0.62, stroke="#b5a89c")
    dog_thumbnail(parts, gen_ix, gen_iy, size=32, tint="#f2dfaa")
    caption_tile(parts, gen_tx, gen_ty, ["a dog"], fill="#f6f0df", stroke="#e6cfc1")
    parts.append(
        f'<text x="{gen_ix:.1f}" y="{gen_iy + 28:.1f}" text-anchor="middle" font-size="9" '
        f'font-weight="900" fill="#c9a84a">generic image</text>'
    )

    # Hierarchy depth bracket between fine and generic samples.
    parts.append(
        '<path d="M52 188V328" fill="none" stroke="#1d76bb" stroke-width="1.6" '
        'stroke-linecap="round" marker-start="url(#softArrow)" marker-end="url(#softArrow)"/>'
    )
    parts.append(
        '<text x="38" y="262" text-anchor="middle" font-size="9" font-weight="850" fill="#1d76bb" '
        'transform="rotate(-90 38 262)">hyperbolic depth</text>'
    )

    # Level guides.
    parts.append(
        '<path d="M118 188H352" stroke="#d9cec1" stroke-width="1" stroke-dasharray="4 5" opacity=".85"/>'
    )
    parts.append(
        '<path d="M148 328H322" stroke="#d9cec1" stroke-width="1" stroke-dasharray="4 5" opacity=".85"/>'
    )

    panel_footer(
        parts,
        34,
        418,
        402,
        "Fine caption and image sit on opposite flanks, higher on the sheet",
        "Generic caption and image anchor near the nadir, farther from the apex",
        "#e7f0e9",
        "#cbdccd",
    )
    parts.append("</g>")


def semantic_svg() -> str:
    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" role="img">',
        SVG_TEXT_STYLE,
        "<defs>",
        '<linearGradient id="sphereFill" x1="0" y1="0" x2="1" y2="1">'
        '<stop offset="0" stop-color="#ffffff"/><stop offset="1" stop-color="#eef3f8"/></linearGradient>',
        '<linearGradient id="hyperFill" x1="0" y1="0" x2="0" y2="1">'
        '<stop offset="0" stop-color="#eef3f8"/><stop offset="0.52" stop-color="#e7f0e9"/>'
        '<stop offset="1" stop-color="#ffffff"/></linearGradient>',
        '<marker id="softArrow" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="6" '
        'markerHeight="6" orient="auto-start-reverse">'
        '<path d="M 0 0 L 10 5 L 0 10 z" fill="#1d76bb"/></marker>',
        "</defs>",
        f'<rect width="{WIDTH}" height="{HEIGHT}" rx="30" fill="#ffffff"/>',
        f'<rect x="26" y="26" width="{WIDTH - 52}" height="{HEIGHT - 52}" rx="24" fill="#ffffff" stroke="#c5d9e8"/>',
        '<text x="590" y="58" text-anchor="middle" font-size="26" font-weight="900" '
        f'fill="{COLORS["ink"]}">Transitioning from Euclidean to Hyperbolic</text>',
    ]
    euclidean_panel(parts)
    matcha_bridge(parts)
    hyperbolic_panel(parts)

    legend_y = 586
    parts.extend(
        [
            f'<rect x="52" y="{legend_y}" width="286" height="28" rx="12" fill="#ffffff" stroke="#c5d9e8"/>',
            f'<rect x="68" y="{legend_y + 8}" width="12" height="12" rx="3" fill="{COLORS["image"]}"/>',
            f'<text x="86" y="{legend_y + 18}" font-size="10" font-weight="800" fill="#000000">image embedding</text>',
            f'<circle cx="214" cy="{legend_y + 14}" r="6" fill="{COLORS["text"]}"/>',
            f'<text x="228" y="{legend_y + 18}" font-size="10" font-weight="800" fill="#000000">text embedding</text>',
        ]
    )
    caption_lines: list[str] = []
    caption_inner = 268
    hyper_cx = 674 + 470 / 2
    for paragraph in (
        "Hyperbolic lifting places fine image and caption on opposite flanks, higher on the sheet.",
        "Generic pairs anchor near the nadir (e.g., a dog), separated from fine-grained samples.",
    ):
        caption_lines.extend(wrap_lines(paragraph, caption_inner, 10, weight="900"))
    tight_multiline_box(
        parts,
        caption_lines,
        cx=hyper_cx,
        y=610,
        size=10,
        weight="900",
        fill="#e7f0e9",
        stroke="#cbdccd",
        pad_x=10,
        pad_y=6,
        line_gap=12,
        rx=10,
    )
    parts.append("</svg>")
    return "".join(parts)


def svg_data_uri(svg: str) -> str:
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("ascii")


def patch_index(svg: str) -> None:
    html_text = INDEX.read_text(encoding="utf-8")
    marker = '<div class="hero-visual fade"><img src="'
    start = html_text.find(marker)
    if start < 0:
        raise RuntimeError("hero-visual image marker not found in index.html")
    src_start = start + len(marker)
    src_end = html_text.find('"', src_start)
    html_text = html_text[:src_start] + svg_data_uri(svg) + html_text[src_end:]
    INDEX.write_text(html_text, encoding="utf-8")


def main() -> None:
    svg = semantic_svg()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(svg, encoding="utf-8")
    patch_index(svg)
    print(f"Wrote {OUT}")
    print(f"Patched {INDEX}")


if __name__ == "__main__":
    main()
