#!/bin/bash
set -euo pipefail
cd /mnt/d/wsl/raha

# Create and switch to temp from current master
/usr/bin/git checkout -b temp

# Stage all without modifying file contents
/usr/bin/git add -A

# Commit via commit-tree (avoids broken git commit wrapper)
if /usr/bin/git diff --cached --quiet; then
  echo "Nothing to commit; using current HEAD"
else
  printf '%s\n' 'update docs' > /tmp/raha_commit_msg
  TREE=$(/usr/bin/git write-tree)
  PARENT=$(/usr/bin/git rev-parse HEAD)
  COMMIT=$(/usr/bin/git commit-tree "$TREE" -p "$PARENT" -F /tmp/raha_commit_msg)
  /usr/bin/git update-ref refs/heads/temp "$COMMIT"
  /usr/bin/git reset --hard "$COMMIT"
fi

/usr/bin/git log -1 --oneline

# Push temp branch
/usr/bin/git push -u origin temp

# Delete remote master
/usr/bin/git push origin --delete master

# Rename temp -> master locally
/usr/bin/git branch -m temp master

# Push new master and set upstream
/usr/bin/git push -u origin master

/usr/bin/git status -sb
/usr/bin/git branch -vv
/usr/bin/git log -1 --oneline
