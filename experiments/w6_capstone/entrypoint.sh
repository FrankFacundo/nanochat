#!/bin/bash
# W6 - Parameter Golf: the single reproducible entrypoint.
#
# One command, no arguments, no interactive choices. It reads the frozen protocol,
# enforces the sealed split, times the run itself, and refuses to proceed if the
# protocol has drifted. This file IS the candidate: locking the candidate means
# committing this file and tagging it.
#
#   bash experiments/w6_capstone/entrypoint.sh            # the timed 30-minute run
#   bash experiments/w6_capstone/entrypoint.sh --final    # + the ONE sealed-test eval
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
export PYTHONUNBUFFERED=1

PROTOCOL=experiments/w6_capstone/protocol.json
[ -f "$PROTOCOL" ] || { echo "FATAL: protocol.json missing. Run 01_freeze_protocol.py first."; exit 1; }

# 1) verify the freeze (fails loudly if data/tokenizer/objective/timer drifted)
python -m experiments.w6_capstone.01_freeze_protocol | tee experiments/w6_capstone/logs/entrypoint_freeze_check.log
grep -q "FREEZE VIOLATION" experiments/w6_capstone/logs/entrypoint_freeze_check.log && {
  echo "FATAL: protocol drift detected. Results under the new conditions are not comparable."; exit 1; }

# 2) enforce the sealed split for the timed run - train AND val exclude these shards
export NANOCHAT_SEALED_SHARDS="$(python -c "import json;print(','.join(json.load(open('$PROTOCOL'))['data']['sealed_test_shards']))")"
export NANOCHAT_SEED=42
echo "Sealed shards: $NANOCHAT_SEALED_SHARDS"

# 3) THE CANDIDATE. Every tuned value lives here and nowhere else.
#    Locked value: --num-iterations is chosen IN ADVANCE to fit 30 minutes; it is never
#    adjusted after seeing the clock (that would be selection on the outcome).
STAMP=$(date -u +%Y%m%d-%H%M%S)
LOG=experiments/w6_capstone/logs/candidate_${STAMP}.log
START=$(date +%s)
python -m scripts.base_train \
  --depth=6 --aspect-ratio=64 --head-dim=64 --window-pattern=L --max-seq-len=512 \
  --device-batch-size=32 --total-batch-size=16384 \
  --num-iterations=3000 \
  --embedding-lr=0.6 --unembedding-lr=0.016 --matrix-lr=0.04 --scalar-lr=1.0 \
  --eval-every=250 --eval-tokens=131072 --core-metric-every=-1 --sample-every=-1 \
  --save-every=-1 --model-tag=capstone --run=dummy 2>&1 | tee "$LOG"
END=$(date +%s)
ELAPSED=$((END-START))
echo "WALL CLOCK: ${ELAPSED}s ($(python -c "print(f'{$ELAPSED/60:.2f}')") min)" | tee -a "$LOG"
if [ "$ELAPSED" -gt 1800 ]; then
  echo "*** RUN VOID: exceeded the 1800s budget. Do NOT report this number. ***" | tee -a "$LOG"
  exit 2
fi
echo "FINAL val_bpb: $(grep 'Validation bpb' "$LOG" | tail -1)" | tee -a "$LOG"

# 4) the ONE-TIME sealed test, only with --final, only after the candidate is locked
if [ "${1:-}" = "--final" ]; then
  SEAL=experiments/w6_capstone/logs/SEALED_TEST_OPENED
  [ -f "$SEAL" ] && { echo "FATAL: the sealed test has already been opened ($(cat $SEAL)).
It is now a second validation split and must be relabelled. Refusing."; exit 3; }
  date -u +%Y-%m-%dT%H:%M:%SZ > "$SEAL"
  unset NANOCHAT_SEALED_SHARDS
  python -m experiments.w6_capstone.03_sealed_test --model-tag=capstone 2>&1 \
    | tee experiments/w6_capstone/logs/sealed_test_${STAMP}.log
fi
