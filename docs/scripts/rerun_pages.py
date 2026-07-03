#!/usr/bin/env python3
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

creds = Path.home() / ".git-credentials"
text = creds.read_text(encoding="utf-8", errors="ignore")
match = re.search(r"https://[^:]*:([^@]+)@github\.com", text)
if not match:
    sys.exit("NO_TOKEN")

token = match.group(1)
headers = {
    "Authorization": "token {}".format(token),
    "Accept": "application/vnd.github+json",
    "User-Agent": "raha-pages-rerun",
}
STUCK_RUN = 28636318731
FAILED_RUN = 28636136897


def api(path, method="GET", data=None):
    body = None if data is None else json.dumps(data).encode("utf-8")
    req = urllib.request.Request(
        "https://api.github.com{}".format(path),
        data=body,
        headers=headers,
        method=method,
    )
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.status, resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8")


code, body = api("/repos/andyj1/raha/actions/runs/{}/cancel".format(STUCK_RUN), "POST")
print("cancel stuck run {}: HTTP {}".format(STUCK_RUN, code))
print(body)

code, body = api("/repos/andyj1/raha/actions/runs/{}/rerun".format(FAILED_RUN), "POST")
print("full rerun failed run {}: HTTP {}".format(FAILED_RUN, code))
print(body)

code, body = api("/repos/andyj1/raha/actions/runs?per_page=3")
runs = json.loads(body)["workflow_runs"]
for run in runs:
    print(
        "{id} #{num} {status:10} {concl:10} {url}".format(
            id=run["id"],
            num=run["run_number"],
            status=run["status"],
            concl=run.get("conclusion") or "-",
            url=run["html_url"],
        )
    )
