#!/usr/bin/env python3
"""Patch analysis and classification sections in index.html."""
from __future__ import annotations

import base64
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "index.html"
RES = ROOT / "res"

CLASSIFICATION_CSS = """    .classification-layout { grid-template-columns: minmax(0, 1.45fr) minmax(280px, .55fr); align-items: start; }
    .classification-layout .table-shell { overflow-x: hidden; }
    .classification-layout table { table-layout: fixed; min-width: 0; width: 100%; }
    .classification-layout th, .classification-layout td { padding: 8px 6px; font-size: .84rem; white-space: normal; word-break: break-word; }
    .classification-samples-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; padding: 10px; background: #ffffff; align-items: stretch; }
    .figure.classification-samples { overflow: visible; }
    .classification-sample-cell { min-width: 0; border: 1px solid #c5d9e8; border-radius: 8px; background: #ffffff; display: flex; flex-direction: column; height: 100%; }
    .classification-sample-cell .qual-label { border-bottom: 1px solid #c5d9e8; text-align: center; font-size: .76rem; padding: 8px 4px; flex: 0 0 auto; white-space: normal; line-height: 1.2; min-height: 2.6em; display: flex; align-items: center; justify-content: center; }
    .classification-sample-cell .media { flex: 1 1 auto; display: grid; place-items: center; min-height: 220px; padding: 10px 8px; background: #ffffff; overflow: hidden; border-radius: 0 0 8px 8px; }
    .figure.classification-samples .classification-sample-cell .media img { width: auto; max-width: 100%; height: auto; max-height: 220px; object-fit: contain; object-position: center center; display: block; margin: 0 auto; }"""

ANALYSIS_GRID = """        <div class="analysis-layout">
          <div class="figure fade analysis-fig2"><div class="media"><img src="__FIG2__" alt="Fig. 2 component ablation chart."></div><div class="caption">Fig. 2: Component attribution under the same protocol. Range matching provides the main relevance-transfer gain, while residual control improves the full model as a refinement when anchored by the range.</div></div>
          <div class="analysis-a6-column">
            <div class="figure fade analysis-a6-panel"><div class="media"><img src="__FIGA6_AB__" alt="Fig. A6 panels (a) and (b)."></div></div>
            <div class="figure fade analysis-a6-bottom"><div class="media"><img src="__FIGA6_CD__" alt="Fig. A6 panels (c) and (d)."></div><div class="caption">Fig. A6: Sensitivity and ablation analysis for matching terms, geometry, and residual control.</div></div>
          </div>
        </div>"""

ANALYSIS_CSS = """    .analysis-layout { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 16px; align-items: start; }
    .analysis-fig2 { min-width: 0; display: flex; flex-direction: column; }
    .analysis-fig2 .media { overflow: visible; width: 100%; }
    .analysis-fig2 .media img { width: 100%; height: auto; object-fit: contain; }
    .analysis-fig2 .caption { align-self: center; width: fit-content; max-width: 100%; margin-inline: auto; box-sizing: border-box; text-wrap: pretty; }
    .analysis-a6-column { display: grid; grid-template-rows: auto auto; gap: 10px; align-items: start; min-width: 0; }
    .analysis-a6-column .figure { min-width: 0; }
    .analysis-a6-panel .media, .analysis-a6-bottom .media { background: #ffffff; overflow: hidden; line-height: 0; padding: 0; }
    .analysis-a6-panel .media img, .analysis-a6-bottom .media img { display: block; width: 100%; max-width: 100%; height: auto; object-fit: contain; object-position: center center; }
    .analysis-a6-bottom .caption { border-top: 1px solid rgba(29,118,187,.12); }"""

CLASSIFICATION_QUAL_GRID = """
        <div class="figure-gap"></div>
        <div class="qual-grid classification-qual-grid">
          <div class="figure fade"><div class="qual-label">CIFAR-100</div><div class="media"><img src="__QUAL_CIFAR__" alt="CIFAR-100 qualitative distilled samples from the appendix."></div><div class="caption">Distilled data comparison on CIFAR-100. Left: initial samples. Middle: Baseline-distilled. Right: RAHA-distilled.</div></div>
          <div class="figure fade"><div class="qual-label">CUB-200-2011</div><div class="media"><img src="__QUAL_CUB__" alt="CUB qualitative distilled samples from the appendix."></div><div class="caption">Distilled data comparison on CUB-200-2011. Left: initial samples. Middle: Baseline-distilled. Right: RAHA-distilled.</div></div>
          <div class="figure fade"><div class="qual-label">Stanford Cars</div><div class="media"><img src="__QUAL_CARS__" alt="Stanford Cars qualitative distilled samples from the appendix."></div><div class="caption">Distilled data comparison on Stanford Cars. Left: initial samples. Middle: Baseline-distilled. Right: RAHA-distilled.</div></div>
          <div class="figure fade"><div class="qual-label">ImageNet-1K</div><div class="media"><img src="__QUAL_IMAGENET__" alt="ImageNet qualitative distilled samples from the appendix."></div><div class="caption">Distilled data comparison on ImageNet-1K. Left: initial samples. Middle: Baseline-distilled. Right: RAHA-distilled.</div></div>
        </div>"""

LEAD_OLD = "At 100 pairs, RAHA improves top-1 prompted classification on all four benchmarks reported in the appendix. RAHA-distilled 2×2 samples are shown beside the accuracy table."
LEAD_NEW = "At 100 pairs, RAHA improves top-1 prompted classification on all four benchmarks reported in the appendix. RAHA-distilled 2×2 samples are shown beside the accuracy table, with full per-dataset comparisons below."


def data_uri(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def patch_analysis_css(html: str) -> str:
    patterns = [
        re.compile(
            r"    \.analysis-a6-stack \{.*?\n    \.analysis-a6 \.caption \{.*?\}\n",
            re.S,
        ),
        re.compile(
            r"    \.analysis-layout \{.*?\n    \.analysis-a6-bottom \.caption \{.*?\}\n",
            re.S,
        ),
    ]
    for pattern in patterns:
        if pattern.search(html):
            html = pattern.sub(ANALYSIS_CSS + "\n", html, count=1)
            return html
    insert_at = html.find("    .qual-grid, .analysis-grid {")
    if insert_at >= 0 and ".analysis-layout" not in html:
        html = html[:insert_at] + ANALYSIS_CSS + "\n" + html[insert_at:]
    return html


def patch_analysis(html: str, fig2_uri: str, figa6_ab_uri: str, figa6_cd_uri: str) -> str:
    section_start = html.find('id="analysis"')
    section_end = html.find("</section>", section_start)
    grid_start = html.find('<div class="analysis-layout">', section_start, section_end)
    if grid_start < 0:
        grid_start = html.find('<div class="analysis-grid">', section_start, section_end)
    if grid_start < 0:
        raise RuntimeError("analysis-grid not found")

    depth = 0
    pos = grid_start
    while pos < section_end:
        open_idx = html.find("<div", pos, section_end)
        close_idx = html.find("</div>", pos, section_end)
        if close_idx < 0 or (open_idx >= 0 and open_idx < close_idx):
            depth += 1
            pos = open_idx + 4
        else:
            depth -= 1
            pos = close_idx + 6
            if depth == 0:
                grid_end = pos
                break
    else:
        raise RuntimeError("could not find end of analysis-grid")

    grid = (
        ANALYSIS_GRID.replace("__FIG2__", fig2_uri)
        .replace("__FIGA6_AB__", figa6_ab_uri)
        .replace("__FIGA6_CD__", figa6_cd_uri)
    )
    html = html[:grid_start] + grid + html[grid_end:]
    html = patch_analysis_css(html)

    media_old = ".qual-grid, .analysis-grid { grid-template-columns: 1fr; } .analysis-layout { grid-template-columns: 1fr; } .analysis-top-right, .analysis-a6-bottom { grid-column: 1; } .analysis-top-right { grid-template-columns: 1fr; } .classification-samples-grid"
    media_new = ".qual-grid, .analysis-grid { grid-template-columns: 1fr; } .analysis-layout { grid-template-columns: 1fr; } .classification-samples-grid"
    if media_old in html:
        html = html.replace(media_old, media_new)

    html = html.replace(
        "    .analysis-stack { display: grid; grid-template-columns: 1fr; gap: 16px; }\n", ""
    )
    return html


def patch_classification_css(html: str) -> str:
    css_start = html.find("    .classification-layout {")
    css_end = html.find("    .figure-gap {", css_start)
    if css_start >= 0 and css_end > css_start:
        return html[:css_start] + CLASSIFICATION_CSS + "\n" + html[css_end:]
    return html


def patch_classification_qual(html: str, qual_uris: dict[str, str]) -> str:
    grid = CLASSIFICATION_QUAL_GRID
    for key, uri in qual_uris.items():
        grid = grid.replace(f"__{key.upper()}__", uri)

    section_start = html.find('id="classification"')
    section_end = html.find("</section>", section_start)
    block = html[section_start:section_end]

    if "classification-qual-grid" in block:
        qual_start = html.find('<div class="qual-grid classification-qual-grid">', section_start, section_end)
        depth = 0
        pos = qual_start
        while pos < section_end:
            open_idx = html.find("<div", pos, section_end)
            close_idx = html.find("</div>", pos, section_end)
            if close_idx < 0 or (open_idx >= 0 and open_idx < close_idx):
                depth += 1
                pos = open_idx + 4
            else:
                depth -= 1
                pos = close_idx + 6
                if depth == 0:
                    qual_end = pos
                    break
        else:
            raise RuntimeError("could not find end of classification-qual-grid")
        html = html[:qual_start] + grid.strip() + html[qual_end:]
        return html

    marker = '</tbody></table></div>\n        </div>'
    insert_at = html.find(marker, section_start, section_end)
    if insert_at < 0:
        raise RuntimeError("classification layout close marker not found")
    insert_at += len(marker)
    return html[:insert_at] + grid + html[insert_at:]


def patch_setup_embed(html: str) -> str:
    setup = ROOT / "generated_figures" / "setup.svg"
    if not setup.exists():
        return html
    uri = "data:image/svg+xml;base64," + base64.b64encode(setup.read_bytes()).decode("ascii")
    marker = '<div class="figure setup-figure fade"><img src="'
    start = html.find(marker)
    if start < 0:
        return html
    src_start = start + len(marker)
    src_end = html.find('"', src_start)
    return html[:src_start] + uri + html[src_end:]


def patch(html: str) -> str:
    fig2_uri = data_uri(RES / "fig2_ablation.png")
    figa6_ab_uri = data_uri(RES / "fig_a6_ab.png")
    figa6_cd_uri = data_uri(RES / "fig_a6_cd.png")
    qual_uris = {
        "qual_cifar": data_uri(RES / "qual_cifar.png"),
        "qual_cub": data_uri(RES / "qual_cub.png"),
        "qual_cars": data_uri(RES / "qual_cars.png"),
        "qual_imagenet": data_uri(RES / "qual_imagenet.png"),
    }

    html = patch_analysis(html, fig2_uri, figa6_ab_uri, figa6_cd_uri)
    html = patch_classification_qual(html, qual_uris)
    html = patch_classification_css(html)
    html = patch_setup_embed(html)
    html = html.replace(LEAD_OLD, LEAD_NEW)
    return html


def main() -> None:
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "build_setup_figure.py")],
        check=True,
        cwd=ROOT,
    )
    for path in (
        RES / "fig2_ablation.png",
        RES / "fig_a6_ab.png",
        RES / "fig_a6_cd.png",
        RES / "qual_cifar.png",
        RES / "qual_cub.png",
        RES / "qual_cars.png",
        RES / "qual_imagenet.png",
    ):
        if not path.exists():
            raise SystemExit(f"Missing crop: {path}. Run build_pdf_crops.py first.")
    INDEX.write_text(patch(INDEX.read_text(encoding="utf-8")), encoding="utf-8")
    print(f"Patched {INDEX}")


if __name__ == "__main__":
    main()
