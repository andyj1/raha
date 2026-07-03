#!/bin/bash
set -euo pipefail
cd /mnt/d/wsl/raha
/usr/bin/git add -A
printf '%s\n' 'update docs' > /tmp/raha_commit_msg
TREE=$(/usr/bin/git write-tree)
COMMIT=$(/usr/bin/git commit-tree "$TREE" -p HEAD -F /tmp/raha_commit_msg)
/usr/bin/git update-ref refs/heads/master "$COMMIT"
/usr/bin/git log -1 --oneline
/usr/bin/git status -sb
