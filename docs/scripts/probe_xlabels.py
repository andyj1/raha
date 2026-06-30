#!/usr/bin/env python3
from pathlib import Path
import subprocess

GEN = Path("/mnt/d/wsl/raha/docs/generated_figures")
p14 = GEN / "inspect_p14_160.png"
p30 = GEN / "inspect_p30_160.png"

def crop(page, geom, out, trim=False):
    cmd = ["convert", str(page), "-crop", geom, "+repage"]
    if trim:
        cmd.extend(["-fuzz", "12%", "-trim", "+repage"])
    cmd.append(str(out))
    subprocess.run(cmd, check=True)

print("Fig2 no trim:")
for y, h in [(232, 210), (232, 220), (234, 215), (234, 225), (236, 210), (236, 220)]:
    o = GEN / f"probe_f2nt_{y}_{h}.png"
    crop(p14, f"780x{h}+295+{y}", o)
    print(f"  y={y} h={h}: {o.stat().st_size//1024}KB")

print("A6 no trim:")
for y, h in [(248, 108), (250, 110), (252, 112), (254, 115), (256, 118)]:
    oab = GEN / f"probe_abnt_{y}_{h}.png"
    ocd = GEN / f"probe_cdnt_{y}_{h}.png"
    crop(p30, f"416x{h}+264+{y}", oab)
    crop(p30, f"416x{h}+680+{y}", ocd)
    print(f"  y={y} h={h}: ab={oab.stat().st_size//1024}KB cd={ocd.stat().st_size//1024}KB")
