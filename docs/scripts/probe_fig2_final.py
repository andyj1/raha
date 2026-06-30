#!/usr/bin/env python3
from __future__ import annotations
import subprocess
from pathlib import Path

PAGE = Path("/mnt/d/wsl/raha/docs/generated_figures/inspect_p14_160.png")
OUT = Path("/mnt/d/wsl/raha/docs/generated_figures")
for y, h in [(285, 245), (288, 242), (290, 240), (275, 255)]:
    geom = f"785x{h}+292+{y}"
    out = OUT / f"probe_fig2_{y}_{h}.png"
    subprocess.run(["convert", str(PAGE), "-crop", geom, "+repage", str(out)], check=True)
    print(geom, out.stat().st_size // 1024, "KB")
