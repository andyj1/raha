#!/usr/bin/env python3
from __future__ import annotations
import subprocess
from pathlib import Path

PAGE = Path("/mnt/d/wsl/raha/docs/generated_figures/inspect_p30_160.png")
OUT = Path("/mnt/d/wsl/raha/docs/generated_figures")

pairs = [
    ("a6_ab_half", "416x98+264+202"),
    ("a6_cd_half", "416x98+680+202"),
    ("a6_ab_half2", "400x100+270+200"),
    ("a6_cd_half2", "400x100+690+200"),
    ("a6_row_full", "832x100+264+200"),
]
for name, geom in pairs:
    out = OUT / f"probe_{name}.png"
    subprocess.run(["convert", str(PAGE), "-crop", geom, "+repage", str(out)], check=True)
    print(name, geom, out.stat().st_size // 1024, "KB")
