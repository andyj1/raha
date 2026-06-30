#!/usr/bin/env python3
from pathlib import Path
import subprocess

RES = Path("/mnt/d/wsl/raha/docs/res")
TMP = Path("/mnt/d/wsl/raha/docs/generated_figures")

for name in ["fig_a6_ab", "fig_a6_cd"]:
    src = RES / f"{name}.png"
    for fuzz in [8, 12, 15, 20]:
        out = TMP / f"{name}_trim{fuzz}.png"
        subprocess.run(
            ["convert", str(src), "-fuzz", f"{fuzz}%", "-trim", "+repage", str(out)],
            check=True,
        )
        r = subprocess.run(["identify", "-format", "%wx%h", str(out)], capture_output=True, text=True)
        print(name, fuzz, r.stdout.strip())
