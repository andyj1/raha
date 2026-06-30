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
        f'fill="{COLORS["ink"]}">Euclidean space</text>'
    )
    parts.append(
        '<text x="219" y="52" text-anchor="middle" font-size="10.5" fill="#000000">'
        "fine pairs face each other; generic pair sits between them</text>"
    )
    parts.append('<text x="54" y="78" font-size="10" font-weight="900" fill="#78a99a">image</text>')
    parts.append('<text x="372" y="78" font-size="10" font-weight="900" fill="#9fb7d9">text</text>')

    parts.append('<circle cx="219" cy="244" r="104" fill="url(#sphereFill)" stroke="#c5d9e8" stroke-width="1.8"/>')
    for ry in (30, 52, 74):
        parts.append(
            f'<ellipse cx="219" cy="244" rx="104" ry="{ry}" fill="none" stroke="#e1d6ca" stroke-width="1.4"/>'
        )

    pairs = [
        ("fine", 104, 182, 334, 166, "yellow dog", 146),
        ("fine", 120, 316, 318, 300, "brown dog", 334),
        ("generic", 176, 244, 262, 244, "dog", 268),
    ]
    for kind, ix, iy, tx, ty, label, pill_y in pairs:
        fill_i = COLORS["generic"] if kind == "generic" else COLORS["image"]
        fill_t = COLORS["generic"] if kind == "generic" else COLORS["text"]
        pair_link(parts, ix, iy, tx, ty, opacity=0.55 if kind == "generic" else 0.72)
        node(parts, ix, iy, "I", fill_i)
        node(parts, tx, ty, "T", fill_t)
        pill(
            parts,
            (ix + tx) / 2,
            pill_y,
            f"{kind}: {label}",
            "#eef3f8" if kind != "generic" else "#f6f0df",
            "#cfdae8" if kind != "generic" else "#e6cfc1",
            size=9.5,
        )

    panel_footer(
        parts,
        28,
        418,
        382,
        "Flat geometry mixes all directions",
        "each pair uses one direct contrastive link",
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


def hyperbolic_panel(parts: list[str]) -> None:
    parts.append(f'<g transform="translate(674,{PANEL_TOP})">')
    parts.append('<rect x="0" y="0" width="470" height="468" rx="18" fill="#f7f1e8" stroke="#c5d9e8"/>')
    parts.append(
        '<text x="235" y="34" text-anchor="middle" font-size="17" font-weight="900" '
        f'fill="{COLORS["ink"]}">Hyperbolic space</text>'
    )
    parts.append(
        '<text x="235" y="52" text-anchor="middle" font-size="10.5" fill="#000000">'
        "generic pair at nadir; fine pairs higher on opposite flanks</text>"
    )
    parts.append(
        '<path d="M136 104C158 140 176 182 192 226C205 264 216 300 235 338C254 300 265 264 278 226C294 182 312 140 334 104" '
        'fill="url(#hyperFill)" stroke="#c5d9e8" stroke-width="1.8" opacity=".84"/>'
    )
    for cy, rx, ry in ((116, 92, 20), (156, 74, 16), (196, 56, 12), (236, 38, 9), (282, 22, 6), (326, 9, 3.5)):
        parts.append(
            f'<ellipse cx="235" cy="{cy}" rx="{rx}" ry="{ry}" fill="none" stroke="#d9cec1" stroke-width="1.4"/>'
        )
    parts.append(
        '<path d="M235 338C222 292 210 246 196 202C186 170 174 144 158 120" fill="none" '
        'stroke="#e5dad0" stroke-width="1.6"/>'
    )
    parts.append(
        '<path d="M235 338C248 292 262 246 276 202C286 170 298 144 314 120" fill="none" '
        'stroke="#e5dad0" stroke-width="1.6"/>'
    )
    parts.append('<text x="148" y="78" font-size="10" font-weight="900" fill="#78a99a">image flank</text>')
    parts.append('<text x="318" y="78" font-size="10" font-weight="900" fill="#9fb7d9">text flank</text>')

    pairs = [
        ("generic", 214, 338, 256, 338, "dog", 368),
        ("fine", 164, 228, 306, 200, "yellow dog", 186),
        ("fine", 148, 168, 322, 140, "brown dog", 146),
    ]
    for kind, ix, iy, tx, ty, label, pill_y in pairs:
        fill_i = COLORS["generic"] if kind == "generic" else COLORS["image"]
        fill_t = COLORS["generic"] if kind == "generic" else COLORS["text"]
        pair_link(parts, ix, iy, tx, ty, opacity=0.58 if kind == "generic" else 0.78)
        node(parts, ix, iy, "I", fill_i)
        node(parts, tx, ty, "T", fill_t)
        pill(
            parts,
            (ix + tx) / 2,
            pill_y,
            f"{kind}: {label}",
            "#eef3f8" if kind != "generic" else "#f6f0df",
            "#cfdae8" if kind != "generic" else "#e6cfc1",
            size=9.5,
        )

    covariance_decomposition(
        parts,
        164,
        228,
        306,
        200,
        box_x=300,
        box_y=248,
        box_w=158,
        title="cross-covariance split",
    )
    panel_footer(
        parts,
        34,
        418,
        402,
        "Hierarchy separates generic from fine pairs",
        "range/residual still come from image-text cross-covariance",
        "#e7f0e9",
        "#cbdccd",
    )
    parts.append('<path d="M235 92V354" stroke="#c8baad" stroke-width="1.8" stroke-dasharray="5 6"/>')
    parts.append(
        '<text x="226" y="236" text-anchor="end" font-size="10" font-weight="800" fill="#3f4d57">time-like</text>'
    )
    parts.append(
        '<text x="226" y="250" text-anchor="end" font-size="10" font-weight="800" fill="#3f4d57">axis</text>'
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
        "In hyperbolic RAHA, dominant range and residual decompose cross-covariance for one pair.",
        "Not slope or endpoint distance along the geometry.",
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
