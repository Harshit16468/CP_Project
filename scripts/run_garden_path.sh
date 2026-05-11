#!/usr/bin/env bash
# scripts/run_garden_path.sh
# ===========================
# Run the garden-path entropy/surprisal dissociation analysis.
#
# Prerequisites:
#   - Dundee validation already run (dundee_validation.py)
#     Dundee parquets with GPT-2 metrics at /tmp/psycholingu_dundee/processed
#
# Usage:
#   bash scripts/run_garden_path.sh [--min-regions N]

set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-config.yaml}"
DUNDEE_PROC="${DUNDEE_PROC:-/tmp/psycholingu_dundee/processed}"
OUT_DIR="${OUT_DIR:-/tmp/psycholingu_gp}"
MIN_REGIONS="${MIN_REGIONS:-15}"

echo "============================================================"
echo "  Garden-Path Dissociation Analysis"
echo "  dundee_processed=$DUNDEE_PROC"
echo "  output=$OUT_DIR"
echo "  min_regions=$MIN_REGIONS"
echo "============================================================"

python scripts/garden_path_analysis.py \
    --config  "$CONFIG" \
    --dundee-processed "$DUNDEE_PROC" \
    --out-dir          "$OUT_DIR" \
    --min-regions      "$MIN_REGIONS"

echo ""
echo "Key outputs:"
echo "  $OUT_DIR/metrics/garden_path_dissociation_table.csv"
echo "  $OUT_DIR/figures/garden_path_dissociation.png"
echo ""
echo "Check the dissociation table:"
echo "  Expected: β_entropy[onset] > β_entropy[disambig]"
echo "  Expected: β_surprisal[disambig] > β_surprisal[onset]"
