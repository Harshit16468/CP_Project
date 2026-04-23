#!/bin/bash
#SBATCH -J "PSYCHOLINGU"
#SBATCH -c 4
#SBATCH -G 1
#SBATCH --mem-per-cpu=18000
#SBATCH -o output_3.out
#SBATCH --time="4-00:00:00"
#SBATCH -w gnode077

echo "Time at entrypoint: $(date)"
echo "Working directory: ${PWD}"

# Steps 1-3, 5 are fully cached in /tmp — only rerun 4 (T5 attention) and 6 (Bayesian)
python3 pipeline.py --steps 4,6

echo "--- Pipeline done, running hypothesis analysis --- $(date)"
python3 scripts/run_analysis.py --results /tmp/psycholingu/results/metrics

echo "--- Running extended analysis (H6 + BERT L0H10) --- $(date)"
python3 scripts/extended_analysis.py \
    --model-path /tmp/psycholingu/results/metrics/06_bayes_full.nc \
    --data-path  /tmp/psycholingu/data/processed/05_integration_cost.parquet \
    --out-dir    /tmp/psycholingu/results/figures \
    --n-sentences 150

echo "--- Copying figures to project directory --- $(date)"
cp /tmp/psycholingu/results/figures/*.png figures/figures/ 2>/dev/null || true
cp /tmp/psycholingu/results/metrics/*.csv  results/metrics/  2>/dev/null || true
cp /tmp/psycholingu/results/metrics/*.nc   results/metrics/  2>/dev/null || true

echo "Time at exit: $(date)"
