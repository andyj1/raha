#!/usr/bin/env python3
import json
import re
import subprocess
import urllib.request
from pathlib import Path

token = re.search(
    r"https://[^:]*:([^@]+)@github\.com",
    (Path.home() / ".git-credentials").read_text(encoding="utf-8", errors="ignore"),
).group(1)
req = urllib.request.Request(
    "https://api.github.com/repos/andyj1/raha",
    data=json.dumps({"default_branch": "master"}).encode("utf-8"),
    headers={
        "Authorization": "token {}".format(token),
        "Accept": "application/vnd.github+json",
        "Content-Type": "application/json",
    },
    method="PATCH",
)
with urllib.request.urlopen(req, timeout=60) as resp:
    data = json.loads(resp.read().decode("utf-8"))
    print("default_branch:", data["default_branch"])

subprocess.check_call(["/usr/bin/git", "-C", "/mnt/d/wsl/raha", "push", "origin", "--delete", "temp"])
subprocess.check_call(["/usr/bin/git", "-C", "/mnt/d/wsl/raha", "status", "-sb"])
subprocess.check_call(["/usr/bin/git", "-C", "/mnt/d/wsl/raha", "branch", "-vv"])
subprocess.check_call(["/usr/bin/git", "-C", "/mnt/d/wsl/raha", "log", "-1", "--oneline"])
