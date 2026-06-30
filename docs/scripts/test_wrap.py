#!/usr/bin/env python3
import importlib.util

spec = importlib.util.spec_from_file_location("bp", "/mnt/d/wsl/raha/docs/scripts/build_page.py")
bp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bp)

text = (
    "Train the same vision-language model faster, then evaluate retrieval, "
    "transfer, and prompted classification."
)
for mw in [700, 580, 520, 480]:
    lines = bp._wrap_text_lines(text, mw - 36, 14)
    size = 14 * bp.FIGURE_FONT_SCALE
    cw = size * 0.52
    content_w = max(len(line) * cw for line in lines)
    box_w = min(mw, content_w + 36)
    print(f"max_width={mw} box_w={box_w:.1f} lines={len(lines)}")
    for line in lines:
        print(f"  [{len(line):2d}] {line}")
