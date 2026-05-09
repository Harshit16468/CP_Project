#!/bin/bash
#SBATCH -J "PSYCHOLINGU_NS_LEX"
#SBATCH -c 4
#SBATCH -G 1
#SBATCH --mem-per-cpu=18000
#SBATCH -o output_ns_lexical.out
#SBATCH --time="04:00:00"
#SBATCH -w gnode061

echo "Time at entrypoint: $(date)"
echo "Working directory: ${PWD}"

# Step 6 only — runs the 5 new lexical variants. Existing 9 .nc files are
# detected by the cache check and skipped.
python3 pipeline.py --config config.yaml --steps 6

# Copy new outputs back
mkdir -p results/metrics results/figures
cp -nv /tmp/psycholingu/results/metrics/06_bayes_*.nc       results/metrics/   2>/dev/null || true
cp -nv /tmp/psycholingu/results/metrics/06_loo_*.pkl        results/metrics/   2>/dev/null || true
cp -nv /tmp/psycholingu/results/metrics/06_summary_*.csv    results/metrics/   2>/dev/null || true
cp -fv /tmp/psycholingu/results/metrics/06_model_comparison.csv  results/metrics/  2>/dev/null || true
cp -fv /tmp/psycholingu/results/figures/*.png  results/figures/  2>/dev/null || true

echo "Time at exit: $(date)"
echo "NS lexical .nc files now in results/metrics/"
ls results/metrics/06_bayes_*.nc | sort
