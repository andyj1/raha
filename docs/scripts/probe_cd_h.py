#!/usr/bin/env python3
from pathlib import Path
import subprocess
GEN = Path("/mnt/d/wsl/raha/docs/generated_figures")
p30 = GEN / "inspect_p30_160.png"
for h in [200, 205, 210, 215]:
    subprocess.run(["convert", str(p30), "-crop", f"416x{h}+680+235", "+repage", "-strip", str(GEN / f"cd_h{h}.png")], check=True)
