#!/usr/bin/env python3
import json
import re
import urllib.request
from pathlib import Path

token = re.search(
    r"https://[^:]*:([^@]+)@github\.com",
    (Path.home() / ".git-credentials").read_text(encoding="utf-8", errors="ignore"),
).group(1)
headers = {"Authorization": "token {}".format(token), "Accept": "application/vnd.github+json"}
req = urllib.request.Request(
    "https://api.github.com/repos/andyj1/raha/actions/runs?per_page=5",
    headers=headers,
)
runs = json.loads(urllib.request.urlopen(req, timeout=60).read().decode("utf-8"))["workflow_runs"]
for run in runs:
    print(
        "#{} {} {} {} {}".format(
            run["run_number"],
            run["status"],
            run.get("conclusion") or "-",
            run["head_sha"][:7],
            run["html_url"],
        )
    )
