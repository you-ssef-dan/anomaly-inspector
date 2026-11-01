#!/usr/bin/env bash
# The last step of the retrain_patchcore DAG, in the training image, on the retrain/<time> branch
# that already holds the new data: commit the pipeline's results, upload the data and models to
# DagsHub, push the branch and open a pull request. Then CI checks the models with the quality
# gate, a person merges, and a release deploys.
#
#   scripts/open_retrain_pr.sh '{"bottle": true}'      # the register step's last line
#
# Needs GH_TOKEN (fine-grained: Contents and Pull requests on this repository) and the DagsHub
# token as AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY.
set -euo pipefail
promoted=${1:?the register step result, like {\"bottle\": true\}}
branch=$(git branch --show-current)

git add dvc.lock reports
# A retry finds this commit already made: every step below is safe to run twice.
git diff --cached --quiet || git commit --quiet -m "retrain: models trained on the new confirmed photos"
dvc push --quiet
# The token reaches git through a credential helper, never in a URL or an argument logs could show.
# shellcheck disable=SC2016  # $GH_TOKEN is expanded by the helper's shell, on purpose
git -c credential.helper='!f() { echo username=x-access-token; echo "password=$GH_TOKEN"; }; f' \
  push --quiet origin "HEAD:refs/heads/$branch"

python - "$branch" "$promoted" "$(git log -1 --format=%s HEAD~1)" "$(git remote get-url origin)" <<'EOF'
import json
import os
import re
import sys
import urllib.request

branch, promoted, data_commit, origin = sys.argv[1], json.loads(sys.argv[2]), sys.argv[3], sys.argv[4]
repo = re.search(r"github\.com[/:](.+?)(\.git)?$", origin).group(1)
winners = [c for c, won in promoted.items() if won]
others = [c for c, won in promoted.items() if not won] or ["none"]
body = f"""Opened by the `retrain_patchcore` Airflow DAG.

- Data: {data_commit}
- New champion in MLflow: {", ".join(winners)}
- Retrained, still challenger only: {", ".join(others)}

model-ci checks these models with the quality gate (they are in DVC already, so nothing retrains).
Merge it, then publish a release to deploy.
"""
request = urllib.request.Request(
    f"https://api.github.com/repos/{repo}/pulls",
    data=json.dumps(
        {"title": f"retrain: a better {', '.join(winners)} model", "head": branch, "base": "main", "body": body}
    ).encode(),
    headers={
        "Authorization": f"Bearer {os.environ['GH_TOKEN']}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    },
    method="POST",
)
with urllib.request.urlopen(request) as response:
    print(json.load(response)["html_url"])
EOF
