#!/usr/bin/env python3
from __future__ import annotations
import subprocess
from pathlib import Path

PAGE = Path("/mnt/d/wsl/raha/docs/generated_figures/inspect_p14_160.png")
OUT = Path("/mnt/d/wsl/raha/docs/generated_figures")
for y in range(200, 320, 10):
    for h in (180, 200, 220, 240):
        geom = f"785x{h}+292+{y}"
        out = OUT / f"probe_p14_y{y}_h{h}.png"
        subprocess.run(["convert", str(PAGE), "-crop", geom, "+repage", str(out)], check=True)
        kb = out.stat().st_size // 1024
        if kb > 8:
            print(f"y={y} h={h}: {kb} KB")

PY