#!/bin/bash
set -euo pipefail
cd /mnt/d/wsl/raha

TOKEN=$(python3 - <<'PY'
import re
from pathlib import Path
text = Path.home().joinpath('.git-credentials').read_text(encoding='utf-8', errors='ignore')
print(re.search(r'https://[^:]*:([^@]+)@github\.com', text).group(1))
PY
)

api() {
  local method="$1"
  local path="$2"
  local data="${3:-}"
  if [ -n "$data" ]; then
    curl -sS -X "$method" \
      -H "Authorization: token $TOKEN" \
      -H "Accept: application/vnd.github+json" \
      -H "Content-Type: application/json" \
      -d "$data" \
      "https://api.github.com$path"
  else
    curl -sS -X "$method" \
      -H "Authorization: token $TOKEN" \
      -H "Accept: application/vnd.github+json" \
      "https://api.github.com$path"
  fi
}

echo "Setting default branch to temp..."
api PATCH /repos/andyj1/raha '{"default_branch":"temp"}'
echo

echo "Deleting remote master..."
/usr/bin/git push origin --delete master

echo "Renaming temp -> master locally..."
/usr/bin/git branch -m temp master

echo "Pushing master..."
/usr/bin/git push -u origin master

echo "Restoring default branch to master..."
api PATCH /repos/andyj1/raha '{"default_branch":"master"}'
echo

echo "Deleting remote temp..."
/usr/bin/git push origin --delete temp || true

/usr/bin/git status -sb
/usr/bin/git branch -vv
/usr/bin/git log -1 --oneline
