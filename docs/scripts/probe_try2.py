#!/usr/bin/env python3
from pathlib import Path
import subprocess

GEN = Path("/mnt/d/wsl/raha/docs/generated_figures")
p14 = GEN / "inspect_p14_160.png"
p30 = GEN / "inspect_p30_160.png"

def crop(page, geom, out):
    subprocess.run(
        ["convert", str(page), "-crop", geom, "+repage", "-strip", str(out)],
        check=True,
    )

for y, h in [(228, 240), (230, 238), (232, 235), (234, 232)]:
    crop(p14, f"780x{h}+295+{y}", GEN / f"f2_try_{y}_{h}.png")

for y, h in [(235, 200), (238, 195), (240, 190)]:
    crop(p30, f"416x{h}+264+{y}", GEN / f"ab_try_{y}_{h}.png")
    crop(p30, f"416x{h}+680+{y}", GEN / f"cd_try_{y}_{h}.png")

print("ok")
