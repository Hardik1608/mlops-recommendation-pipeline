#!/usr/bin/env bash
# Shadow deployment, end to end, on a Linux machine with Docker and an NVIDIA GPU:
#   1. full stack on MovieLens 1M: the trainer registers pure ALS (v1, champion) and the
#      hybrid (v2, challenger: 10% of users)
#   2. SASRec is trained on the GPU from the host venv (scripts/gpu.sh setup) and registered
#      in the same MLflow; the gate compares it with the champion on the test window
#   3. a candidate that loses the gate becomes the shadow model: the API hot-swaps it in
#      and scores it on 20% of requests in the background, never serving its output
#   4. a load test, then Prometheus: model-compute P95 per variant and the shadow's
#      agreement (overlap@10) with the list actually served
set -euo pipefail
cd "$(dirname "$0")/.."
export RECSYS_DATASET=ml-1m
PROM=http://localhost:9090/api/v1/query

query() {
  curl -s "$PROM" --data-urlencode "query=$1" | .venv/bin/python -c "
import json, sys
for r in json.load(sys.stdin)['data']['result']:
    print('   ', r['metric'] or '{}', round(float(r['value'][1]), 6))"
}

echo "== 1. stack (build + train ALS champion and hybrid challenger)"
docker compose up -d --build
until [ "$(docker compose ps -a trainer --format '{{.State}}')" = "exited" ]; do sleep 10; done
docker compose ps -a trainer --format '{{.Status}}' | grep -q "Exited (0)" \
  || { docker compose logs trainer | tail -30; echo "trainer failed"; exit 1; }
until curl -fs localhost:8000/ready > /dev/null; do sleep 5; done

echo "== 2. SASRec on the GPU, registered in the stack's MLflow"
export MLFLOW_TRACKING_URI=http://localhost:5000
.venv/bin/recsys train --config configs/experiments/ml-1m-sasrec.yaml --no-export \
  | tee logs/shadow-demo-train.json
# The CLI logs to stdout too: the summary is the JSON block that starts at the last "{" line.
summary() {  # arguments: keys into the JSON summary
  .venv/bin/python - "$@" <<'PY'
import json, sys
lines = open("logs/shadow-demo-train.json", encoding="utf-8").read().splitlines()
start = max(i for i, line in enumerate(lines) if line == "{")
value = json.loads(" ".join(lines[start:]))  # JSON ignores the joining whitespace
for key in sys.argv[1:]:
    value = value[key]
print(value)
PY
}
version=$(summary model_version)
action=$(summary decision action)
echo "SASRec registered as v$version; gate decision: $action"

echo "== 3. shadow alias -> API hot-swap (polls every 30 s)"
.venv/bin/recsys shadow --version "$version"
until curl -s localhost:8000/ready | grep -q '"shadow"'; do sleep 5; done

echo "== 4. load test (120 s at 50 req/s, 10% new users), then Prometheus"
.venv/bin/python -m recsys.loadtest --url http://localhost:8000 --rate 50 --duration 120 \
  --cold-fraction 0.1 --label shadow-demo
echo "model compute P95 by variant (s):"
query 'histogram_quantile(0.95, sum by (le, variant) (rate(recsys_model_compute_seconds_bucket[3m])))'
echo "shadow agreement with the served list (mean overlap@10):"
query 'sum(rate(recsys_shadow_overlap_ratio_sum[3m])) / sum(rate(recsys_shadow_overlap_ratio_count[3m]))'
echo "shadow scoring by result (req/s):"
query 'sum by (result) (rate(recsys_shadow_requests_total[3m]))'
echo "Grafana: http://localhost:3000/d/recsys-overview  (row: Shadow model)"
