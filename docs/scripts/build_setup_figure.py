#!/usr/bin/env python3
"""Build and patch the abstract setup figure in the RAHA project page."""
from __future__ import annotations

import base64
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "generated_figures" / "setup.svg"
INDEX = ROOT / "index.html"
SCRIPTS = Path(__file__).resolve().parent


def load_setup_svg():
    spec = importlib.util.spec_from_file_location("build_page", SCRIPTS / "build_page.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load build_page.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_page"] = module
    spec.loader.exec_module(module)
    return module.setup_svg()


def svg_data_uri(svg: str) -> str:
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("ascii")


def patch_index(svg: str) -> None:
    html_text = INDEX.read_text(encoding="utf-8")
    marker = '<div class="figure setup-figure fade"><img src="'
    start = html_text.find(marker)
    if start < 0:
        raise RuntimeError("setup-figure image marker not found in index.html")
    src_start = start + len(marker)
    src_end = html_text.find('"', src_start)
    html_text = html_text[:src_start] + svg_data_uri(svg) + html_text[src_end:]
    INDEX.write_text(html_text, encoding="utf-8")


def main() -> None:
    svg = load_setup_svg()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(svg, encoding="utf-8")
    patch_index(svg)
    print(f"Wrote {OUT}")
    print(f"Patched {INDEX}")


if __name__ == "__main__":
    main()
