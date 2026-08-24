#!/bin/bash
# W2/C - measure the noise floor: 3 seeds of the pilot control, everything else frozen.
set -e
source .venv/bin/activate
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
export PYTHONUNBUFFERED=1
OUT=experiments/w2_ablations/logs
for SEED in 42 43 44; do
  echo "=== seed $SEED ==="
  NANOCHAT_SEED=$SEED python -m scripts.base_train \
    --depth=6 --aspect-ratio=64 --head-dim=64 --window-pattern=L --max-seq-len=512 \
    --device-batch-size=32 --total-batch-size=16384 --num-iterations=600 \
    --eval-every=100 --eval-tokens=131072 --core-metric-every=-1 --sample-every=-1 \
    --save-every=-1 --model-tag=w2noise --run=dummy 2>&1 | tee "$OUT/noise_seed${SEED}.log" \
    | grep -E "Validation bpb|Total training time|Using non-default seed"
done
echo "ALL DONE"
