#!/usr/bin/env bash
# Smoke test of a running API: it becomes ready, passes a good part and rejects a defective one.
#
#   scripts/smoke_test.sh [base_url] [good_image] [defective_image]
#
# Defaults: the API from `make api-up` (bottle) and two bottle photos from holdout. CI runs it
# against every image it builds (Phase 8).
set -euo pipefail

url=${1:-http://localhost:8000}
good=${2:-data/raw/mvtec_ad/bottle/test/good/001.png}
defective=${3:-data/raw/mvtec_ad/bottle/test/broken_large/002.png}

fail() { echo "FAIL: $1" >&2; exit 1; }
for photo in "$good" "$defective"; do [ -f "$photo" ] || fail "no such photo: $photo"; done

# A container that just started needs a few seconds to load its model: wait up to 60 s.
for _ in $(seq 1 30); do
    curl -fsS "$url/health/ready" > /dev/null 2>&1 && break
    sleep 2
done
curl -fsS "$url/health/ready" > /dev/null || fail "$url is not ready after 60 s"
echo "ok: $url is ready"

# is_anomalous (true or false) for one photo.
decision() {
    curl -fsS -F "file=@$1;type=image/png" "$url/v1/predict" \
        | python3 -c "import json, sys; print(str(json.load(sys.stdin)['is_anomalous']).lower())"
}

[ "$(decision "$good")" = false ] || fail "a good part was rejected: $good"
echo "ok: good part passes ($good)"
[ "$(decision "$defective")" = true ] || fail "a defective part passed: $defective"
echo "ok: defective part rejected ($defective)"
