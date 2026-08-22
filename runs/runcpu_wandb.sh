#!/bin/bash

# Run the Mac/CPU educational pipeline with online Weights & Biases logging.
#
# First-time setup:
#   uv run wandb login
#
# Usage:
#   ./runs/runcpu_wandb.sh
#   ./runs/runcpu_wandb.sh my-run-name
#   WANDB_RUN=my-run-name ./runs/runcpu_wandb.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

if pgrep -f '[r]uns/runcpu.sh' >/dev/null; then
    echo "Another runcpu.sh process is already running. Wait for it to finish before starting this run." >&2
    exit 1
fi

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
RUN_NAME="${1:-${WANDB_RUN:-m3max-d6-$TIMESTAMP}}"
if [ -z "$RUN_NAME" ] || [ "$RUN_NAME" = "dummy" ]; then
    echo "The W&B run name must be non-empty and cannot be 'dummy'." >&2
    exit 1
fi

SAFE_RUN_NAME="$(printf '%s' "$RUN_NAME" | tr -c '[:alnum:]._-' '_')"
LOG_DIR="${NANOCHAT_LOG_DIR:-$HOME/.cache/nanochat/logs}"
LOG_PATH="$LOG_DIR/${SAFE_RUN_NAME}_$TIMESTAMP.log"
mkdir -p "$LOG_DIR"

export WANDB_RUN="$RUN_NAME"
export WANDB_MODE=online

echo "W&B run: $WANDB_RUN"
echo "Local log: $LOG_PATH"
echo "W&B projects: nanochat (pretraining), nanochat-sft (SFT)"

bash runs/runcpu.sh 2>&1 | tee "$LOG_PATH"
