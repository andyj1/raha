#!/usr/bin/env python3
"""Build the 1x4 RAHA-distilled sample grid for the classification section."""
from __future__ import annotations

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
BASE_DPI = 160
PAGE = 26

# Right-hand (RAHA-distilled) column of each appendix qualitative figure.
RAHA_CROPS = [
    ("qual_cifar_raha.png", "280x256+830+226", "CIFAR-100"),
    ("qual_cub_raha.png", "280x224+830+568", "CUB-200-2011"),
    ("qual_cars_raha.png", "280x203+830+872", "Stanford Cars"),
    ("qual_imagenet_raha.png", "280x228+830+1124", "ImageNet-1K"),
]


def scale_geom(geometry: str, factor: float) -> str:
    parts = geometry.replace("+", " +").split()
    w, h = [int(float(parts[0].split("x")[0])), int(float(parts[0].split("x")[1]))]
    x = int(float(parts[1].lstrip("+")))
    y = int(float(parts[2].lstrip("+")))
    return f"{round(w * factor)}x{round(h * factor)}+{round(x * factor)}+{round(y * factor)}"


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, cwd=ROOT)


def render_page(page: int, out: Path) -> None:
    run([
        "gs", "-dNOSAFER", "-dNOPAUSE", "-dBATCH", "-sDEVICE=pngalpha",
        f"-r{RENDER_DPI}", f"-dFirstPage={page}", f"-dLastPage={page}",
        f"-sOutputFile={out}", str(PDF),
    ])


CANVAS_W = 525
CANVAS_H = 480


def crop(src: Path, geometry: str, out: Path) -> None:
    run([
        "convert", str(src), "-crop", geometry, "+repage",
        "-fuzz", "1%", "-trim", "+repage",
        "-background", "white", "-gravity", "center",
        "-extent", f"{CANVAS_W}x{CANVAS_H}",
        "-strip", "-quality", "98", str(out),
    ])


def patch_index(panel_paths: list[Path], labels: list[str]) -> None:
    html_path = ROOT / "index.html"
    html = html_path.read_text(encoding="utf-8")

    classification_css = """    .classification-layout { grid-template-columns: minmax(0, 1.45fr) minmax(280px, .55fr); align-items: start; }
    .classification-layout .table-shell { overflow-x: hidden; }
    .classification-layout table { table-layout: fixed; min-width: 0; width: 100%; }
    .classification-layout th, .classification-layout td { padding: 8px 6px; font-size: .84rem; white-space: normal; word-break: break-word; }
    .classification-samples-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; padding: 10px; background: #ffffff; align-items: stretch; }
    .figure.classification-samples { overflow: visible; }
    .classification-sample-cell { min-width: 0; border: 1px solid #c5d9e8; border-radius: 8px; background: #ffffff; display: flex; flex-direction: column; height: 100%; }
    .classification-sample-cell .qual-label { border-bottom: 1px solid #c5d9e8; text-align: center; font-size: .76rem; padding: 8px 4px; flex: 0 0 auto; white-space: normal; line-height: 1.2; min-height: 2.6em; display: flex; align-items: center; justify-content: center; }
    .classification-sample-cell .media { flex: 1 1 auto; display: grid; place-items: center; min-height: 220px; padding: 10px 8px; background: #ffffff; overflow: hidden; border-radius: 0 0 8px 8px; }
    .figure.classification-samples .classification-sample-cell .media img { width: auto; max-width: 100%; height: auto; max-height: 220px; object-fit: contain; object-position: center center; display: block; margin: 0 auto; }"""
    css_start = html.find("    .classification-layout {")
    css_end = html.find("    .figure-gap {", css_start)
    if css_start >= 0 and css_end > css_start:
        html = html[:css_start] + classification_css + "\n" + html[css_end:]

    layout_marker = '<div class="figure-table-grid classification-layout">'
    layout_start = html.find(layout_marker)
    section_start = html.find('id="classification"')
    section_end = html.find("</section>", section_start)
    if layout_start < 0:
        raise RuntimeError("classification layout marker not found")

    table_start = html.find('<div class="table-shell fade"><table>', layout_start, section_end)
    table_end = html.find("</table></div>", table_start) + len("</table></div>")
    layout_end = html.find("\n        </div>", table_end) + len("\n        </div>")
    table_html = html[table_start:table_end]

    cells = "\n".join(
        f"""            <div class="classification-sample-cell">
              <div class="qual-label">{label}</div>
              <div class="media"><img src="{path.relative_to(ROOT).as_posix()}" alt="RAHA-distilled samples on {label}."></div>
            </div>"""
        for path, label in zip(panel_paths, labels)
    )
    new_block = f"""        {layout_marker}
          <div class="figure classification-samples fade">
            <div class="classification-samples-grid">
{cells}
            </div>
          </div>
          {table_html}
        </div>
"""
    html = html[:layout_start] + new_block + html[layout_end:]

    html_path.write_text(html, encoding="utf-8")
    print(f"Patched {html_path}")


def main() -> None:
    if not PDF.exists():
        raise SystemExit(f"PDF not found: {PDF}")
    if not shutil.which("gs") or not shutil.which("convert"):
        raise SystemExit("Ghostscript and ImageMagick convert are required")

    factor = RENDER_DPI / BASE_DPI
    RES.mkdir(parents=True, exist_ok=True)
    GEN.mkdir(parents=True, exist_ok=True)

    page_path = GEN / f"page_{PAGE}_{RENDER_DPI}.png"
    if not page_path.exists():
        print(f"Rendering page {PAGE} at {RENDER_DPI} dpi...")
        render_page(PAGE, page_path)

    panel_paths: list[Path] = []
    labels: list[str] = []
    for out_name, geom, label in RAHA_CROPS:
        scaled = scale_geom(geom, factor)
        target = RES / out_name
        print(f"Cropping {out_name}: {scaled}")
        crop(page_path, scaled, target)
        shutil.copy2(target, GEN / out_name)
        panel_paths.append(target)
        labels.append(label)
        print(f"  -> {target} ({target.stat().st_size // 1024} KB)")

    patch_index(panel_paths, labels)


if __name__ == "__main__":
    main()
