#!/usr/bin/env python3
from __future__ import annotations
import subprocess
from pathlib import Path

PAGE = Path("/mnt/d/wsl/raha/docs/generated_figures/inspect_p30_160.png")
OUT = Path("/mnt/d/wsl/raha/docs/generated_figures")
for h in (98, 105, 112, 118):
    for y in (200, 202, 205):
        geom_ab = f"416x{h}+264+{y}"
        geom_cd = f"416x{h}+680+{y}"
        out_ab = OUT / f"probe_ab_{y}_{h}.png"
        out_cd = OUT / f"probe_cd_{y}_{h}.png"
        subprocess.run(["convert", str(PAGE), "-crop", geom_ab, "+repage", str(out_ab)], check=True)
        subprocess.run(["convert", str(PAGE), "-crop", geom_cd, "+repage", str(out_cd)], check=True)
        print(f"y={y} h={h}: ab={out_ab.stat().st_size//1024}KB cd={out_cd.stat().st_size//1024}KB")
