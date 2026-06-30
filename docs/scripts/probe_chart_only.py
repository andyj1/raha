#!/usr/bin/env python3
"""Find chart-only crops for analysis figures."""
from __future__ import annotations

import subprocess
from pathlib import Path

GEN = Path("/mnt/d/wsl/raha/docs/generated_figures")

def crop(page: Path, geom: str, out: Path) -> int:
    subprocess.run(["convert", str(page), "-crop", geom, "+repage", str(out)], check=True)
    return out.stat().st_size // 1024

def main() -> None:
    p14 = GEN / "inspect_p14_160.png"
    p30 = GEN / "inspect_p30_160.png"

    print("=== Fig 2 page 14 ===")
    for y in range(228, 248, 2):
        for h in (165, 175, 185, 195):
            name = f"fig2_y{y}_h{h}"
            kb = crop(p14, f"785x{h}+292+{y}", GEN / f"probe_{name}.png")
            if kb >= 8:
                print(f"  {name}: {kb} KB")

    print("=== Fig A6 page 30 ab ===")
    for y in range(200, 218, 2):
        for h in (72, 80, 88, 96):
            name = f"a6ab_y{y}_h{h}"
            kb = crop(p30, f"416x{h}+264+{y}", GEN / f"probe_{name}.png")
            if kb >= 2:
                print(f"  {name}: {kb} KB")

    print("=== Fig A6 page 30 cd ===")
    for y in range(200, 218, 2):
        for h in (72, 80, 88, 96):
            name = f"a6cd_y{y}_h{h}"
            kb = crop(p30, f"416x{h}+680+{y}", GEN / f"probe_{name}.png")
            if kb >= 2:
                print(f"  {name}: {kb} KB")

if __name__ == "__main__":
    main()
