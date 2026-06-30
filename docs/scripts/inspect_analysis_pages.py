#!/usr/bin/env python3
"""Inspect PDF pages for Fig. 2 and Fig. A6 placement."""
from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "paper.pdf"
GEN = ROOT / "generated_figures"
GEN.mkdir(exist_ok=True)


def main() -> None:
    for page in range(27, 32):
        out = GEN / f"inspect_p{page}_160.png"
        if not out.exists() or out.stat().st_size < 10000:
            subprocess.run(
                [
                    "gs",
                    "-dNOSAFER",
                    "-dNOPAUSE",
                    "-dBATCH",
                    "-sDEVICE=pngalpha",
                    "-r160",
                    f"-dFirstPage={page}",
                    f"-dLastPage={page}",
                    f"-sOutputFile={out}",
                    str(PDF),
                ],
                check=True,
            )
        print(f"page {page}: {out} ({out.stat().st_size // 1024} KB)")

    text_out = GEN / "pages_27_31.txt"
    subprocess.run(
        ["pdftotext", "-f", "27", "-l", "31", str(PDF), str(text_out)],
        check=False,
    )
    if text_out.exists():
        print(text_out.read_text(encoding="utf-8", errors="replace")[:4000])


if __name__ == "__main__":
    main()
