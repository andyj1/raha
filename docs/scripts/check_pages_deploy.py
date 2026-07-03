#!/usr/bin/env python3
import base64
import re
import urllib.request

url = "https://andyj1.github.io/raha/"
html = urllib.request.urlopen(url, timeout=30).read().decode("utf-8", errors="replace")
print("page bytes:", len(html))
alt = 'alt="Motivation diagram showing LoRS and MERU feeding into RAHA."'
if alt in html:
    pos = html.find(alt)
    img_start = html.rfind('<img src="', 0, pos)
    src = html[img_start + len('<img src="'): html.find('"', img_start + len('<img src="'))]
    svg = base64.b64decode(src.split(",", 1)[1]).decode("utf-8")
    meru = re.search(r"MERU <tspan[^>]*>\(ICML \d{4}\)</tspan>", svg)
    print("live MERU:", meru.group(0) if meru else "not found")
else:
    print("motivation figure not found")

# pages deployment info
for api in [
    "https://api.github.com/repos/andyj1/raha/pages",
    "https://api.github.com/repos/andyj1/raha/pages/builds?per_page=5",
]:
    try:
        req = urllib.request.Request(api, headers={"Accept": "application/vnd.github+json"})
        data = urllib.request.urlopen(req, timeout=20).read().decode()
        print("\n", api)
        print(data[:1200])
    except Exception as e:
        print("\n", api, "->", e)
