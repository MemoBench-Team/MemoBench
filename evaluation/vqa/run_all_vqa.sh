#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT="$SCRIPT_DIR/llm-vqa.py"
LOG_DIR="vqa_logs"
mkdir -p "$LOG_DIR"

MODELS=(
    "LingBot-World"
    "Wan2.2"
    "FantasyWorld"
    "Matrix-Game2"
    "StableVirtualCamera"
    "Open-SoRA"
    "LTX-Video"
    "CogVideoX"
)

for MODEL in "${MODELS[@]}"; do
    LOG="$LOG_DIR/${MODEL}.log"
    echo "========================================"
    echo "Starting: $MODEL  →  $LOG"
    echo "========================================"
    python "$SCRIPT" \
        --model-name "$MODEL" \
        --skip-existing \
        2>&1 | tee "$LOG"
    echo ""
    echo "Finished: $MODEL"
    echo ""
done

echo "All models done."
