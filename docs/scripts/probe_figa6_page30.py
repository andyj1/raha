#!/usr/bin/env python3
"""Probe Fig. A6 split crops on page 30."""
from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "generated_figures"
PDF = ROOT / "paper.pdf"
PAGE_IMG = GEN / "inspect_p30_160.png"

if not PAGE_IMG.exists():
    subprocess.run(
        [
            "gs", "-dNOSAFER", "-dNOPAUSE", "-dBATCH", "-sDEVICE=pngalpha",
            "-r160", "-dFirstPage=30", "-dLastPage=30",
            f"-sOutputFile={PAGE_IMG}", str(PDF),
        ],
        check=True,
    )

CANDIDATES = [
    ("a6p30_ab_a", "832x95+264+218"),
    ("a6p30_ab_b", "832x100+264+212"),
    ("a6p30_ab_c", "832x105+264+208"),
    ("a6p30_cd_a", "832x95+264+318"),
    ("a6p30_cd_b", "832x100+264+312"),
    ("a6p30_cd_c", "832x105+264+308"),
    ("a6p30_full", "832x210+264+208"),
]


def main() -> None:
    for name, geom in CANDIDATES:
        out = GEN / f"probe_{name}.png"
        subprocess.run(
            ["convert", str(PAGE_IMG), "-crop", geom, "+repage", str(out)],
            check=True,
        )
        print(f"{name} {geom}: {out.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
