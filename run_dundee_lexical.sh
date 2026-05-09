#!/bin/bash
#SBATCH -J "PSYCHOLINGU_DUNDEE_LEX"
#SBATCH -c 4
#SBATCH -G 1
#SBATCH --mem-per-cpu=18000
#SBATCH -o output_dundee_lexical.out
#SBATCH --time="04:00:00"
#SBATCH -w gnode061

echo "Time at entrypoint: $(date)"
echo "Working directory: ${PWD}"

DUNDEE_OUT=/tmp/psycholingu_dundee
NS_METRICS=/home2/ishaan.romil/CP_Project/results/metrics

# Step 6 only with new Dundee variants (existing 9 .nc cached → skipped).
python3 scripts/dundee_validation.py \
    --config     config.yaml \
    --steps      6 \
    --out-dir    $DUNDEE_OUT \
    --ns-metrics $NS_METRICS

# Copy new outputs
mkdir -p figures/dundee results/dundee_metrics
cp -nv $DUNDEE_OUT/figures/*.png    figures/dundee/           2>/dev/null || true
cp -nv $DUNDEE_OUT/metrics/*.csv    results/dundee_metrics/   2>/dev/null || true
cp -nv $DUNDEE_OUT/metrics/*.nc     results/dundee_metrics/   2>/dev/null || true
cp -nv $DUNDEE_OUT/metrics/*.pkl    results/dundee_metrics/   2>/dev/null || true

echo "Time at exit: $(date)"
echo "Dundee new variants in results/dundee_metrics/"
ls results/dundee_metrics/dundee_bayes_*.nc | sort
