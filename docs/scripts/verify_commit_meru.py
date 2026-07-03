#!/usr/bin/env python3
import subprocess
svg = subprocess.check_output(
    ["/usr/bin/git", "-C", "/mnt/d/wsl/raha", "show", "1165161:docs/res/motivation.svg"],
    text=True,
)
import re
m = re.search(r"MERU <tspan[^>]*>\(ICML \d{4}\)</tspan>", svg)
print("committed:", m.group(0) if m else "missing")
