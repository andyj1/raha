#!/usr/bin/env python3
"""Find and probe analysis figure pages in the RAHA PDF."""
from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "paper.pdf"
GEN = ROOT / "generated_figures"
GEN.mkdir(exist_ok=True)

# Candidate pages from prior tuning and appendix numbering
PAGES = [13, 14, 15, 16, 28, 29, 30, 31]


def render(page: int, dpi: int = 160) -> Path:
    out = GEN / f"inspect_p{page}_{dpi}.png"
    subprocess.run(
        [
            "gs", "-dNOSAFER", "-dNOPAUSE", "-dBATCH", "-sDEVICE=pngalpha",
            f"-r{dpi}", f"-dFirstPage={page}", f"-dLastPage={page}",
            f"-sOutputFile={out}", str(PDF),
        ],
        check=True,
    )
    return out


def crop_preview(page_img: Path, geometry: str, out: Path) -> None:
    subprocess.run(
        ["convert", str(page_img), "-crop", geometry, "+repage", str(out)],
        check=True,
    )


def main() -> None:
    for page in PAGES:
        img = render(page)
        print(f"rendered page {page}: {img.stat().st_size // 1024} KB")

    page29 = GEN / "inspect_p29_160.png"
    # Current geometries at 160 DPI
    crops = {
        "fig2_current": "832x244+264+234",
        "fig_a6_ab_current": "832x100+264+241",
        "fig_a6_cd_current": "832x105+264+339",
        "fig_a6_full": "832x205+264+241",
    }
    for name, geom in crops.items():
        out = GEN / f"probe_{name}.png"
        crop_preview(page29, geom, out)
        print(f"  {name}: {geom} -> {out.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
