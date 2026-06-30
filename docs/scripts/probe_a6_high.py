#!/usr/bin/env python3
from pathlib import Path
import subprocess

GEN = Path("/mnt/d/wsl/raha/docs/generated_figures")
p30 = GEN / "inspect_p30_160.png"

print("A6 higher y:")
for y, h in [(240, 70), (244, 72), (248, 74), (252, 76), (256, 78), (260, 80)]:
    oab = GEN / f"probe_ab_{y}_{h}.png"
    ocd = GEN / f"probe_cd_{y}_{h}.png"
    subprocess.run(["convert", str(p30), "-crop", f"416x{h}+264+{y}", "+repage", str(oab)], check=True)
    subprocess.run(["convert", str(p30), "-crop", f"416x{h}+680+{y}", "+repage", str(ocd)], check=True)
    print(f"  y={y} h={h}: ab={oab.stat().st_size//1024}KB cd={ocd.stat().st_size//1024}KB")
