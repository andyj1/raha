#!/usr/bin/env python3
"""Extract high-quality figure crops from the RAHA paper PDF."""
from __future__ import annotations

import base64
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "res"
GEN = ROOT / "generated_figures"
PDF = ROOT / "RAHA_final.pdf"
if not PDF.exists():
    PDF = ROOT / "paper.pdf"

RENDER_DPI = 300
BASE_DPI = 160  # crop geometries in build_page.py were tuned at this DPI


def scale_geom(geometry: str, factor: float) -> str:
    parts = geometry.replace("+", " +").split()
    w, h = [int(float(parts[0].split("x")[0])), int(float(parts[0].split("x")[1]))]
    x = int(float(parts[1].lstrip("+")))
    y = int(float(parts[2].lstrip("+")))
    return f"{round(w * factor)}x{round(h * factor)}+{round(x * factor)}+{round(y * factor)}"


CROPS = {
    "fig2_ablation.png": (14, "780x240+295+228"),
    "fig_a6_ab.png": (30, "416x200+264+235"),
    "fig_a6_cd.png": (30, "416x210+680+235"),
    "fig_a6_sensitivity.png": (30, "832x200+264+235"),
    "main_qualitative.png": (12, "785x230+292+606"),
    "qual_cifar.png": (26, "840x256+270+226"),
    "qual_cub.png": (26, "840x224+270+568"),
    "qual_cars.png": (26, "840x216+270+856"),
    "qual_imagenet.png": (26, "840x228+270+1124"),
}

def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, cwd=ROOT)


def render_page(page: int, out: Path) -> None:
    run([
        "gs", "-dNOSAFER", "-dNOPAUSE", "-dBATCH", "-sDEVICE=pngalpha",
        f"-r{RENDER_DPI}", f"-dFirstPage={page}", f"-dLastPage={page}",
        f"-sOutputFile={out}", str(PDF),
    ])


def crop(
    src: Path,
    geometry: str,
    out: Path,
    *,
    trim: bool = True,
    chart_trim: bool = False,
    border_trim: bool = False,
) -> None:
    cmd = ["convert", str(src), "-crop", geometry, "+repage"]
    if chart_trim:
        cmd.extend(["-fuzz", "12%", "-trim", "+repage"])
    elif border_trim:
        cmd.extend(["-fuzz", "15%", "-trim", "+repage"])
    elif trim:
        cmd.extend(["-fuzz", "1%", "-trim", "+repage"])
    if out.name == "qual_cars.png":
        cmd.extend(["-gravity", "South", "-chop", "0x18", "+repage"])
    cmd.extend(["-strip", "-quality", "98", str(out)])
    run(cmd)


def data_uri(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def replace_img_src(html: str, marker: str, uri: str, *, after: bool = False) -> str:
    idx = html.find(marker)
    if idx < 0:
        raise RuntimeError(f"marker not found: {marker!r}")
    if after:
        src_start = html.find('src="', idx) + 5
    else:
        src_start = html.find('src="', idx) + 5
    src_end = html.find('"', src_start)
    return html[:src_start] + uri + html[src_end:]


ANALYSIS_IMAGES = frozenset({
    "fig2_ablation.png",
    "fig_a6_ab.png",
    "fig_a6_cd.png",
})

A6_BORDER_TRIM = frozenset({
    "fig_a6_ab.png",
    "fig_a6_cd.png",
})


def patch_index(outputs: dict[str, Path]) -> None:
    html = (ROOT / "index.html").read_text(encoding="utf-8")

    analysis_markers = {
        "fig2_ablation.png": 'alt="Fig. 2 component ablation chart."',
        "fig_a6_ab.png": 'alt="Fig. A6 panels (a) and (b)."',
        "fig_a6_cd.png": 'alt="Fig. A6 panels (c) and (d)."',
    }
    for name in ANALYSIS_IMAGES:
        if name not in outputs:
            continue
        marker = analysis_markers[name]
        idx = html.find(marker)
        if idx < 0:
            continue
        src_start = html.rfind('src="', 0, idx) + 5
        src_end = html.find('"', src_start)
        html = html[:src_start] + data_uri(outputs[name]) + html[src_end:]

    embedded_markers = {
        "main_qualitative.png": '<section class="band" id="qualitative">',
        "qual_cifar.png": '<div class="qual-label">CIFAR-100</div><div class="media"><img src="',
        "qual_cub.png": '<div class="qual-label">CUB-200-2011</div><div class="media"><img src="',
        "qual_cars.png": '<div class="qual-label">Stanford Cars</div><div class="media"><img src="',
        "qual_imagenet.png": '<div class="qual-label">ImageNet-1K</div><div class="media"><img src="',
    }
    for name, marker in embedded_markers.items():
        if name not in outputs:
            continue
        idx = html.find(marker)
        if idx < 0:
            continue
        src_start = html.find('src="', idx) + 5
        src_end = html.find('"', src_start)
        html = html[:src_start] + data_uri(outputs[name]) + html[src_end:]

    (ROOT / "index.html").write_text(html, encoding="utf-8")


def main() -> None:
    if not PDF.exists():
        raise SystemExit(f"PDF not found: {PDF}")
    if not shutil.which("gs") or not shutil.which("convert"):
        raise SystemExit("Ghostscript and ImageMagick convert are required")

    factor = RENDER_DPI / BASE_DPI
    RES.mkdir(parents=True, exist_ok=True)
    GEN.mkdir(parents=True, exist_ok=True)

    pages: dict[int, Path] = {}
    outputs: dict[str, Path] = {}
    for out_name, (page, geom) in CROPS.items():
        if page not in pages:
            page_path = GEN / f"page_{page}_{RENDER_DPI}.png"
            if not page_path.exists():
                print(f"Rendering page {page} at {RENDER_DPI} dpi...")
                render_page(page, page_path)
            pages[page] = page_path

        scaled = scale_geom(geom, factor)
        target = RES / out_name
        print(f"Cropping {out_name} from page {page}: {scaled}")
        crop(
            pages[page],
            scaled,
            target,
            trim=out_name not in ANALYSIS_IMAGES,
            chart_trim=False,
            border_trim=out_name in A6_BORDER_TRIM,
        )
        outputs[out_name] = target
        shutil.copy2(target, GEN / out_name)
        print(f"  -> {target} ({target.stat().st_size // 1024} KB)")

    patch_index(outputs)
    print(f"Patched {ROOT / 'index.html'}")


if __name__ == "__main__":
    main()
