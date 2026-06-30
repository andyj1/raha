#!/usr/bin/env python3
from pathlib import Path
import subprocess

GEN = Path("/mnt/d/wsl/raha/docs/generated_figures")
p30 = GEN / "inspect_p30_160.png"

def crop(page, geom, out):
    subprocess.run(["convert", str(page), "-crop", geom, "+repage", str(out)], check=True)

for y, h in [(248, 120), (250, 122), (252, 125), (254, 128), (256, 130)]:
    oab = GEN / f"probe_ab_{y}_{h}.png"
    ocd = GEN / f"probe_cd_{y}_{h}.png"
    crop(p30, f"416x{h}+264+{y}", oab)
    crop(p30, f"416x{h}+680+{y}", ocd)
    print(f"y={y} h={h}")
