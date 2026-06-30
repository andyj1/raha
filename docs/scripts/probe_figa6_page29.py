#!/usr/bin/env python3
"""Probe Fig. A6 split crops on page 29."""
from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "generated_figures" / "inspect_p29_160.png"
OUT = ROOT / "generated_figures"

# Full figure area ~ y 230-450 at 160dpi based on inspect image
CANDIDATES = [
    ("a6_ab_a", "832x88+264+252"),
    ("a6_ab_b", "832x92+264+248"),
    ("a6_ab_c", "832x96+264+244"),
    ("a6_cd_a", "832x92+264+340"),
    ("a6_cd_b", "832x96+264+336"),
    ("a6_cd_c", "832x100+264+332"),
    ("a6_ab_old", "832x100+264+241"),
    ("a6_cd_old", "832x105+264+339"),
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
