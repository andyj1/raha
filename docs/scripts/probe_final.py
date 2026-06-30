#!/usr/bin/env python3
from pathlib import Path
import subprocess

GEN = Path("/mnt/d/wsl/raha/docs/generated_figures")
p30 = GEN / "inspect_p30_160.png"

def crop(page, geom, out):
    subprocess.run(["convert", str(page), "-crop", geom, "+repage", str(out)], check=True)

for y, h in [(252, 135), (254, 138), (256, 140), (258, 142), (260, 145)]:
    ocd = GEN / f"probe_cd_{y}_{h}.png"
    crop(p30, f"416x{h}+680+{y}", ocd)
    print(f"cd y={y} h={h}")

# fig2 final
p14 = GEN / "inspect_p14_160.png"
for y, h in [(232, 220), (234, 225), (236, 228)]:
    crop(p14, f"780x{h}+295+{y}", GEN / f"probe_f2_{y}_{h}.png")
