#!/usr/bin/env python3
from pathlib import Path
import subprocess

GEN = Path("/mnt/d/wsl/raha/docs/generated_figures")
p30 = GEN / "inspect_p30_160.png"

def crop_trim(page, geom, out):
    subprocess.run(
        ["convert", str(page), "-crop", geom, "+repage", "-fuzz", "15%", "-trim", "+repage", "-strip", str(out)],
        check=True,
    )

for y, h in [(240, 198), (243, 195), (245, 192), (247, 190)]:
    crop_trim(p30, f"416x{h}+264+{y}", GEN / f"ab_tight_{y}_{h}.png")
    crop_trim(p30, f"416x{h}+680+{y}", GEN / f"cd_tight_{y}_{h}.png")
    print(y, h)
