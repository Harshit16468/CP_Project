#!/bin/bash
#SBATCH -J "PSYCHOLINGU_DUNDEE"
#SBATCH -c 4
#SBATCH -G 1
#SBATCH --mem-per-cpu=18000
#SBATCH -o output_dundee.out
#SBATCH --time="12:00:00"
#SBATCH -w gnode061

echo "Time at entrypoint: $(date)"
echo "Working directory: ${PWD}"
echo "Node: $(hostname)"

DUNDEE_OUT=/tmp/psycholingu_dundee
NS_METRICS=/home2/ishaan.romil/Psycho_Sanchit/results/metrics

# ── Step 1-3: Load Dundee, compute n-gram + neural surprisal & entropy ───────
echo "--- Step 1-3: Data prep + N-gram + Neural metrics --- $(date)"
python3 scripts/dundee_validation.py \
    --config     config.yaml \
    --steps      1,2,3 \
    --out-dir    $DUNDEE_OUT \
    --ns-metrics $NS_METRICS

# ── Step 5: Integration cost (Stanza UD parsing) ──────────────────────────────
echo "--- Step 5: Integration cost --- $(date)"
python3 scripts/dundee_validation.py \
    --config     config.yaml \
    --steps      5 \
    --out-dir    $DUNDEE_OUT \
    --ns-metrics $NS_METRICS

# ── Step 6: Bayesian modeling (9 variants, LOO-CV) ────────────────────────────
echo "--- Step 6: Bayesian modeling --- $(date)"
python3 scripts/dundee_validation.py \
    --config     config.yaml \
    --steps      6 \
    --out-dir    $DUNDEE_OUT \
    --ns-metrics $NS_METRICS

# ── Cross-dataset comparison plots (no steps flag = plots only) ───────────────
echo "--- Generating cross-dataset comparison plots --- $(date)"
python3 scripts/dundee_validation.py \
    --config     config.yaml \
    --steps      "" \
    --out-dir    $DUNDEE_OUT \
    --ns-metrics $NS_METRICS

# ── Copy outputs back to project directory ────────────────────────────────────
echo "--- Copying outputs to project directory --- $(date)"
mkdir -p figures/dundee  results/dundee_metrics

cp $DUNDEE_OUT/figures/*.png  figures/dundee/             2>/dev/null || true
cp $DUNDEE_OUT/metrics/*.csv  results/dundee_metrics/     2>/dev/null || true
cp $DUNDEE_OUT/metrics/*.nc   results/dundee_metrics/     2>/dev/null || true

echo "Time at exit: $(date)"
echo "Dundee outputs:"
echo "  Figures : $DUNDEE_OUT/figures/"
echo "  Metrics : $DUNDEE_OUT/metrics/"
