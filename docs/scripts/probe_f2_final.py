#!/usr/bin/env python3
from pathlib import Path
import subprocess

GEN = Path("/mnt/d/wsl/raha/docs/generated_figures")
p14 = GEN / "inspect_p14_160.png"

for y, h in [(236, 165), (238, 168), (240, 170), (242, 172)]:
    o = GEN / f"probe_f2_{y}_{h}.png"
    subprocess.run(["convert", str(p14), "-crop", f"780x{h}+295+{y}", "+repage", str(o)], check=True)
    print(f"y={y} h={h}: {o.stat().st_size//1024}KB")
