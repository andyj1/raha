#!/usr/bin/env python3
"""Probe Fig. 2 crop on page 14."""
from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "generated_figures" / "inspect_p14_160.png"
OUT = ROOT / "generated_figures"

# Try several candidate boxes for the ablation bar chart (160 DPI)
CANDIDATES = [
    ("p14_a", "785x230+292+606"),
    ("p14_b", "785x220+292+580"),
    ("p14_c", "800x240+280+590"),
    ("p14_d", "832x244+264+580"),
    ("p14_e", "832x200+264+620"),
    ("p14_f", "700x200+320+600"),
]


def main() -> None:
    for name, geom in CANDIDATES:
        out = OUT / f"probe_{name}.png"
        subprocess.run(
            ["convert", str(PAGE), "-crop", geom, "+repage", str(out)],
            check=True,
        )
        print(f"{name} {geom}: {out.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
