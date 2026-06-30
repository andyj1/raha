#!/usr/bin/env python3
from __future__ import annotations
import subprocess
from pathlib import Path

PAGE = Path("/mnt/d/wsl/raha/docs/generated_figures/inspect_p30_160.png")
OUT = Path("/mnt/d/wsl/raha/docs/generated_figures")
for y in range(180, 280, 5):
    for h in (90, 100, 110, 120):
        geom = f"832x{h}+264+{y}"
        out = OUT / f"probe_p30_y{y}_h{h}.png"
        subprocess.run(["convert", str(PAGE), "-crop", geom, "+repage", str(out)], check=True)
        kb = out.stat().st_size // 1024
        if kb > 8:
            print(f"y={y} h={h}: {kb} KB")
