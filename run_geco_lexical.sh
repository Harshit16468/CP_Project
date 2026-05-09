#!/bin/bash
#SBATCH -J "PSYCHOLINGU_GECO_LEX"
#SBATCH -c 4
#SBATCH -G 1
#SBATCH --mem-per-cpu=18000
#SBATCH -o output_geco_lexical.out
#SBATCH --time="04:00:00"
#SBATCH -w gnode061

echo "Time at entrypoint: $(date)"
echo "Working directory: ${PWD}"

GECO_OUT=/tmp/psycholingu_geco
NS_METRICS=/home2/ishaan.romil/CP_Project/results/metrics

# Step 6 only with new GECO variants (existing 9 .nc cached → skipped).
python3 scripts/geco_validation.py \
    --config     config.yaml \
    --steps      6 \
    --out-dir    $GECO_OUT \
    --ns-metrics $NS_METRICS

# Copy new outputs (pkls + summaries + .nc)
mkdir -p figures/geco results/geco_metrics
cp -nv $GECO_OUT/figures/*.png    figures/geco/             2>/dev/null || true
cp -nv $GECO_OUT/metrics/*.csv    results/geco_metrics/     2>/dev/null || true
cp -nv $GECO_OUT/metrics/*.nc     results/geco_metrics/     2>/dev/null || true
cp -nv $GECO_OUT/metrics/*.pkl    results/geco_metrics/     2>/dev/null || true

echo "Time at exit: $(date)"
echo "GECO new variants in results/geco_metrics/"
ls results/geco_metrics/geco_bayes_*.nc | sort
