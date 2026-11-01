#!/usr/bin/env bash
# Replace the DagsHub token everywhere it is used, from one hidden prompt:
#   1. DVC on this laptop (.dvc/config.local), the only copy on disk
#   2. GitHub: the Actions secret and the Dependabot secret (CI and Dependabot's pull requests)
#   3. Airflow, if it is running: it is restarted with the new token
# Then delete the old token on https://dagshub.com/user/settings/tokens
#
#   scripts/rotate_dagshub_token.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

read -rsp "New DagsHub token (nothing shows while you paste): " token
echo
if [ "${#token}" -lt 20 ]; then
  echo "That is ${#token} characters: not a DagsHub token. Nothing changed." >&2
  exit 1
fi

uv run dvc remote modify --local dagshub access_key_id "$token"
uv run dvc remote modify --local dagshub secret_access_key "$token"
echo "1. DVC on this laptop: updated"

printf '%s' "$token" | gh secret set DAGSHUB_TOKEN
printf '%s' "$token" | gh secret set DAGSHUB_TOKEN --app dependabot
unset token
echo "2. GitHub secrets for Actions and Dependabot: updated"

if [ -n "$(cd airflow && docker compose ps -q airflow-scheduler 2>/dev/null)" ]; then
  make airflow-up > /dev/null
  echo "3. Airflow: restarted with the new token"
else
  echo "3. Airflow is not running: make airflow-up will give it the new token"
fi
echo "Now delete the old token on https://dagshub.com/user/settings/tokens"
