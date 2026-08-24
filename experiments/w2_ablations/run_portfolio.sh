#!/bin/bash
# W2/C - four pilot ablations against the 3-seed control measured in run_noise_floor.sh.
# Every arm: d6, 600 steps, 16384 tok/step, eval_tokens=131072, seed 42. Only the named flag moves.
set -e
source .venv/bin/activate
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
export PYTHONUNBUFFERED=1
export NANOCHAT_SEED=42
OUT=experiments/w2_ablations/logs
COMMON="--depth=6 --aspect-ratio=64 --head-dim=64 --window-pattern=L --max-seq-len=512 \
--device-batch-size=32 --total-batch-size=16384 --num-iterations=600 --eval-every=100 \
--eval-tokens=131072 --core-metric-every=-1 --sample-every=-1 --save-every=-1 --run=dummy"

run () { # name, extra args
  local NAME=$1; shift
  echo "=== $NAME ==="
  python -m scripts.base_train $COMMON --model-tag="w2_$NAME" "$@" 2>&1 | tee "$OUT/arm_$NAME.log" \
    | grep -E "Validation bpb|Total training time"
}
run lr-half   --embedding-lr=0.15 --unembedding-lr=0.004 --matrix-lr=0.01 --scalar-lr=0.25
run lr-double --embedding-lr=0.6  --unembedding-lr=0.016 --matrix-lr=0.04 --scalar-lr=1.0
run wd-zero   --weight-decay=0.0
run tied      --tie-embeddings=1
echo "PORTFOLIO DONE"
