#!/usr/bin/env bash
# Smoke test of a running API.
#
#   scripts/smoke_test.sh [base_url] [good_image] [defective_image]
#   scripts/smoke_test.sh --basic [base_url]
#
# Default: it becomes ready, passes a good part and rejects a defective one (two bottle photos
# from holdout, so it needs the dataset and the real model).
# --basic: for an image built with the tiny random model (CI has no dataset or real model): it
# becomes ready, describes its model, scores a generated photo with its heatmap, and refuses a
# text file. It checks that the image works, not that the model is any good.
set -euo pipefail

basic=false
if [ "${1:-}" = "--basic" ]; then
    basic=true
    shift
fi
url=${1:-http://localhost:8000}
good=${2:-data/raw/mvtec_ad/bottle/test/good/001.png}
defective=${3:-data/raw/mvtec_ad/bottle/test/broken_large/002.png}

fail() { echo "FAIL: $1" >&2; exit 1; }
if [ "$basic" = false ]; then
    for photo in "$good" "$defective"; do [ -f "$photo" ] || fail "no such photo: $photo"; done
fi

# A container that just started needs a few seconds to load its model: wait up to 60 s.
for _ in $(seq 1 30); do
    curl -fsS "$url/health/ready" > /dev/null 2>&1 && break
    sleep 2
done
curl -fsS "$url/health/ready" > /dev/null || fail "$url is not ready after 60 s"
echo "ok: $url is ready"

if [ "$basic" = true ]; then
    work=$(mktemp -d)
    trap 'rm -rf "$work"' EXIT

    curl -fsS "$url/v1/model" | python3 -c "
import json, sys
m = json.load(sys.stdin)
assert m['threshold'] > 0 and m['image_size'] > 0 and m['category'], m
print('ok: model', m['category'], 'threshold', m['threshold'])" || fail "/v1/model is wrong"

    # A random 96 px PNG, written with the standard library only.
    python3 - "$work/photo.png" <<'PY'
import random, struct, sys, zlib

random.seed(0)
w = h = 96
raw = b"".join(b"\x00" + bytes(random.randrange(256) for _ in range(w * 3)) for _ in range(h))

def chunk(kind, data):
    body = kind + data
    return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
       + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
open(sys.argv[1], "wb").write(png)
PY
    curl -fsS -F "file=@$work/photo.png;type=image/png" "$url/v1/predict?heatmap=true" \
        | python3 -c "
import json, sys
r = json.load(sys.stdin)
assert isinstance(r['anomaly_score'], float) and isinstance(r['is_anomalous'], bool), r
assert r['heatmap_png_base64'], 'no heatmap'
print('ok: scored a photo', round(r['anomaly_score'], 3), 'with its heatmap')" \
        || fail "/v1/predict is wrong"

    echo "not an image" > "$work/note.txt"
    code=$(curl -s -o /dev/null -w '%{http_code}' -F "file=@$work/note.txt;type=text/plain" "$url/v1/predict")
    [ "$code" = 415 ] || fail "a text file got HTTP $code instead of 415"
    echo "ok: a text file is refused (415)"
    exit 0
fi

# is_anomalous (true or false) for one photo.
decision() {
    curl -fsS -F "file=@$1;type=image/png" "$url/v1/predict" \
        | python3 -c "import json, sys; print(str(json.load(sys.stdin)['is_anomalous']).lower())"
}

[ "$(decision "$good")" = false ] || fail "a good part was rejected: $good"
echo "ok: good part passes ($good)"
[ "$(decision "$defective")" = true ] || fail "a defective part passed: $defective"
echo "ok: defective part rejected ($defective)"
