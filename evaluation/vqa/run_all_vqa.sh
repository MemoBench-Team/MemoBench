#!/bin/bash
set -e

SCRIPT="/VQA/code/llm-vqa.py"
LOG_DIR="/VQA/logs"
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
