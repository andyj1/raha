#!/usr/bin/env python3
import json
import re
import time
import urllib.request
from pathlib import Path

creds = Path.home() / ".git-credentials"
text = creds.read_text(encoding="utf-8", errors="ignore")
token = re.search(r"https://[^:]*:([^@]+)@github\.com", text).group(1)
headers = {"Authorization": "token {}".format(token), "Accept": "application/vnd.github+json"}


def get_run(run_id):
    req = urllib.request.Request(
        "https://api.github.com/repos/andyj1/raha/actions/runs/{}".format(run_id),
        headers=headers,
    )
    return json.loads(urllib.request.urlopen(req, timeout=60).read().decode("utf-8"))


for _ in range(18):
    for run_id in (28636136897, 28636318731):
        run = get_run(run_id)
        print(
            "#{} {} {} {}".format(
                run["run_number"], run_id, run["status"], run.get("conclusion") or "-"
            )
        )
    latest = get_run(28636136897)
    if latest["status"] == "completed":
        jobs = json.loads(
            urllib.request.urlopen(
                urllib.request.Request(
                    "https://api.github.com/repos/andyj1/raha/actions/runs/28636136897/jobs",
                    headers=headers,
                ),
                timeout=60,
            ).read().decode("utf-8")
        )["jobs"]
        for job in jobs:
            print("  {}: {}".format(job["name"], job.get("conclusion")))
        break
    time.sleep(20)
