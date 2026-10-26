#!/usr/bin/env bash
# SASRec experiments on a Linux GPU box (developed on one RTX 3060 12 GB, Ubuntu 22.04).
#
#   scripts/gpu.sh setup                 # uv + Python 3.11 venv + requirements-gpu.txt (no sudo)
#   scripts/gpu.sh data                  # download, checksum and temporally split ml-1m + ml-25m
#   scripts/gpu.sh probe [dataset]       # 3 SASRec epochs: seconds per epoch and peak VRAM
#   scripts/gpu.sh tune <dataset> <name> <model> <param=v1,v2 ...>
#                                        # validation-window grid -> reports/tuning/<dataset>/
#   scripts/gpu.sh bench <dataset> [models]
#                                        # models on the test window -> reports/benchmark_*
#
# Every stage logs to logs/<stage>-<dataset>[-<name>].log. Run long stages inside tmux.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
VENV=.venv
PY="$VENV/bin/python"
RECSYS="$VENV/bin/recsys"
mkdir -p logs

stage="${1:-}"
shift || true

case "$stage" in
  setup)
    command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
    uv venv --python 3.11 "$VENV"
    uv pip install --python "$PY" -r requirements-gpu.txt
    uv pip install --python "$PY" --no-deps -e .
    "$PY" -c "import torch, onnxruntime as ort; print('torch', torch.__version__, '| cuda', \
torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '', \
'| onnxruntime', ort.__version__)"
    ;;
  data)
    for ds in ml-1m ml-25m; do
      "$RECSYS" download --dataset "$ds"
      "$RECSYS" prepare --dataset "$ds"
    done 2>&1 | tee logs/data.log
    sha256sum data/raw/*.zip | tee -a logs/data.log
    ;;
  probe)
    ds="${1:-ml-1m}"
    "$RECSYS" tune --dataset "$ds" --model sasrec --metric warm_ndcg_at_10 \
      --grid epochs=3 eval_every=3 --name "probe-sasrec" 2>&1 | tee "logs/probe-$ds.log"
    ;;
  tune)
    # Model hyper-parameters are selected on warm users (cold users all get the same
    # popularity fallback), as for the ALS stages in reports/tuning/.
    ds="$1"; name="$2"; model="$3"; shift 3
    "$RECSYS" tune --dataset "$ds" --model "$model" --metric "${METRIC:-warm_ndcg_at_10}" \
      --mlflow --name "$name" --grid "$@" 2>&1 | tee "logs/tune-$ds-$name.log"
    ;;
  bench)
    ds="$1"; models="${2:-}"
    "$RECSYS" benchmark --dataset "$ds" --no-mlflow ${models:+--models "$models" --no-random} \
      2>&1 | tee "logs/bench-$ds${models:+-$models}.log"
    ;;
  *)
    sed -n '2,13p' "$0"
    exit 1
    ;;
esac
