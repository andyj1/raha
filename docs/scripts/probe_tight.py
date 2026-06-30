#!/usr/bin/env python3
from pathlib import Path
import subprocess

GEN = Path("/mnt/d/wsl/raha/docs/generated_figures")
p14 = GEN / "inspect_p14_160.png"
p30 = GEN / "inspect_p30_160.png"

print("Fig2:")
for y, h in [(234, 168), (236, 172), (238, 175), (240, 178), (242, 180)]:
    o = GEN / f"probe_f2_{y}_{h}.png"
    subprocess.run(["convert", str(p14), "-crop", f"785x{h}+292+{y}", "+repage", str(o)], check=True)
    print(f"  y={y} h={h}: {o.stat().st_size//1024}KB")

print("A6:")
for y, h in [(218, 78), (222, 82), (226, 85), (230, 88), (234, 90)]:
    oab = GEN / f"probe_ab_{y}_{h}.png"
    ocd = GEN / f"probe_cd_{y}_{h}.png"
    subprocess.run(["convert", str(p30), "-crop", f"416x{h}+264+{y}", "+repage", str(oab)], check=True)
    subprocess.run(["convert", str(p30), "-crop", f"416x{h}+680+{y}", "+repage", str(ocd)], check=True)
    print(f"  y={y} h={h}: ab={oab.stat().st_size//1024}KB cd={ocd.stat().st_size//1024}KB")
