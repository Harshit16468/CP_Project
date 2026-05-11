#!/usr/bin/env bash
# scripts/run_llama.sh
# =====================
# Run LLaMA 3 8B scale-invariance validation on Natural Stories and Dundee.
#
# Prerequisites:
#   1. Main pipeline already run (NS processed parquets at /tmp/psycholingu/)
#   2. Dundee validation already run (Dundee parquets at /tmp/psycholingu_dundee/)
#   3. HuggingFace login:  huggingface-cli login
#   4. GPU with ≥16 GB VRAM recommended (or ≥32 GB RAM for CPU)
#
# Usage:
#   bash scripts/run_llama.sh [--corpus ns|dundee|both] [--steps 3,6]

set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-config.yaml}"
CORPUS="${1:-both}"
STEPS="${2:-3,6}"
OUT_DIR="${OUT_DIR:-/tmp/psycholingu_llama}"
NS_PROC="${NS_PROC:-/tmp/psycholingu/data/processed}"
DUNDEE_PROC="${DUNDEE_PROC:-/tmp/psycholingu_dundee/processed}"

echo "============================================================"
echo "  LLaMA 3 8B Scale-Invariance Validation"
echo "  corpus=$CORPUS  steps=$STEPS"
echo "  output=$OUT_DIR"
echo "============================================================"

python scripts/llama_validation.py \
    --config "$CONFIG" \
    --corpus "$CORPUS" \
    --steps  "$STEPS" \
    --out-dir         "$OUT_DIR" \
    --ns-processed    "$NS_PROC" \
    --dundee-processed "$DUNDEE_PROC"

echo ""
echo "Done. Next: bash scripts/run_garden_path.sh"
