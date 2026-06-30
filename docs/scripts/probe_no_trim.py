#!/usr/bin/env python3
from pathlib import Path
import subprocess

GEN = Path("/mnt/d/wsl/raha/docs/generated_figures")
p14 = GEN / "inspect_p14_160.png"
p30 = GEN / "inspect_p30_160.png"

def crop_no_trim(page, geom, out):
    subprocess.run(
        ["convert", str(page), "-crop", geom, "+repage", "-strip", str(out)],
        check=True,
    )

# Fig 2 candidates (no trim)
for y, h in [(230, 230), (232, 228), (234, 225), (236, 222)]:
    crop_no_trim(p14, f"780x{h}+295+{y}", GEN / f"final_f2_{y}_{h}.png")

# A6 ab/cd (no trim)
for y, h in [(248, 136), (250, 138), (252, 140)]:
    crop_no_trim(p30, f"416x{h}+264+{y}", GEN / f"final_ab_{y}_{h}.png")
    crop_no_trim(p30, f"416x{h}+680+{y}", GEN / f"final_cd_{y}_{h}.png")

print("done")
