#!/usr/bin/env python3
from __future__ import annotations
import subprocess
from pathlib import Path

PAGE = Path("/mnt/d/wsl/raha/docs/generated_figures/inspect_p30_160.png")
OUT = Path("/mnt/d/wsl/raha/docs/generated_figures")
for y in range(210, 225):
    geom_ab = f"416x92+264+{y}"
    geom_cd = f"416x92+680+{y}"
    out_ab = OUT / "probe_ab_tune.png"
    out_cd = OUT / "probe_cd_tune.png"
    subprocess.run(["convert", str(PAGE), "-crop", geom_ab, "+repage", str(out_ab)], check=True)
    subprocess.run(["convert", str(PAGE), "-crop", geom_cd, "+repage", str(out_cd)], check=True)
    print(f"y={y}: ab={out_ab.stat().st_size//1024} cd={out_cd.stat().st_size//1024}")
