#!/usr/bin/env python3
"""Patch motivation figure embed in index.html from res/motivation.svg."""
from __future__ import annotations

import base64
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "index.html"
MOTIVATION = ROOT / "res" / "motivation.svg"


def svg_data_uri(svg: str) -> str:
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("ascii")


def main() -> None:
    svg = MOTIVATION.read_text(encoding="utf-8")
    html = INDEX.read_text(encoding="utf-8")
    alt = 'alt="Motivation diagram showing LoRS and MERU feeding into RAHA."'
    alt_pos = html.find(alt)
    if alt_pos < 0:
        raise RuntimeError("motivation figure alt marker not found in index.html")
    img_start = html.rfind('<img src="', 0, alt_pos)
    if img_start < 0:
        raise RuntimeError("motivation figure img marker not found in index.html")
    src_start = img_start + len('<img src="')
    src_end = html.find('"', src_start)
    html = html[:src_start] + svg_data_uri(svg) + html[src_end:]
    INDEX.write_text(html, encoding="utf-8")
    print(f"Patched motivation figure in {INDEX}")


if __name__ == "__main__":
    main()
