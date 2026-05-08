#!/bin/bash
#SBATCH -J "PSYCHOLINGU_NS_FINAL"
#SBATCH -c 4
#SBATCH -G 1
#SBATCH --mem-per-cpu=18000
#SBATCH -o output_ns_final.out
#SBATCH --time="12:00:00"
#SBATCH -w gnode061

echo "Time at entrypoint: $(date)"
echo "Working directory: ${PWD}"
echo "Node: $(hostname)"

NS_OUT_METRICS=/tmp/psycholingu/results/metrics
NS_OUT_FIGURES=/tmp/psycholingu/results/figures

# Make sure config points at NS dataset
python3 -c "
import yaml
with open('config.yaml') as fh: c = yaml.safe_load(fh)
c['dataset'] = 'natural_stories'
with open('config.yaml','w') as fh: yaml.safe_dump(c, fh, sort_keys=False)
print('config dataset set to natural_stories')
"

# ── NS Steps 1,2,3,5,6 (skip Step 4 attention — already in results/) ─────────
echo "--- NS pipeline steps 1,2,3,5,6 --- $(date)"
python3 pipeline.py --config config.yaml --steps 1,2,3,5,6

# ── Copy NS .nc + summaries back ─────────────────────────────────────────────
echo "--- Copying NS outputs --- $(date)"
mkdir -p results/metrics results/figures
cp -v $NS_OUT_METRICS/06_bayes_*.nc       results/metrics/   2>/dev/null || true
cp -v $NS_OUT_METRICS/06_loo_*.pkl        results/metrics/   2>/dev/null || true
cp -v $NS_OUT_METRICS/06_summary_*.csv    results/metrics/   2>/dev/null || true
cp -v $NS_OUT_METRICS/06_model_comparison.csv  results/metrics/  2>/dev/null || true
cp -v $NS_OUT_FIGURES/*.png               results/figures/   2>/dev/null || true

# ── MCMC robustness across all three datasets ────────────────────────────────
echo "--- MCMC robustness (NS + GECO + Dundee) --- $(date)"
mkdir -p results/mcmc_diagnostics figures/mcmc

for label in ns geco dundee; do
  case $label in
    ns)
      MET=$NS_OUT_METRICS
      ;;
    geco)
      MET=/tmp/psycholingu_geco/metrics
      [ -d "$MET" ] || MET=/home2/ishaan.romil/Psycho_Sanchit/results/geco_metrics
      ;;
    dundee)
      MET=/tmp/psycholingu_dundee/metrics
      [ -d "$MET" ] || MET=/home2/ishaan.romil/Psycho_Sanchit/results/dundee_metrics
      ;;
  esac

  echo ">>> mcmc_robustness on $label  ($MET)"
  python3 scripts/mcmc_robustness.py \
      --results "$MET" \
      --figures /tmp/psycholingu/results/figures/mcmc_$label \
      || echo "   (mcmc_robustness on $label failed — non-fatal, continuing)"

  cp -v /tmp/psycholingu/results/figures/mcmc_$label/*.png  figures/mcmc/      2>/dev/null || true
  cp -v $MET/mcmc_*.csv                                     results/mcmc_diagnostics/  2>/dev/null || true
done

# Re-cross-dataset comparison plots now that NS .nc files exist
echo "--- Regenerating cross-dataset comparison plots (now with NS .nc) --- $(date)"
python3 scripts/geco_validation.py   --config config.yaml --steps "" \
    --out-dir /tmp/psycholingu_geco   --ns-metrics $NS_OUT_METRICS  || true
python3 scripts/dundee_validation.py --config config.yaml --steps "" \
    --out-dir /tmp/psycholingu_dundee --ns-metrics $NS_OUT_METRICS  || true

cp -v /tmp/psycholingu_geco/figures/*.png    figures/geco/    2>/dev/null || true
cp -v /tmp/psycholingu_dundee/figures/*.png  figures/dundee/  2>/dev/null || true
cp -v /tmp/psycholingu_geco/metrics/cross_dataset_comparison.csv          results/geco_metrics/   2>/dev/null || true
cp -v /tmp/psycholingu_dundee/metrics/cross_dataset_comparison_dundee.csv results/dundee_metrics/ 2>/dev/null || true

echo "Time at exit: $(date)"
echo "Outputs:"
echo "  results/metrics/         (NS .nc files + summaries)"
echo "  results/mcmc_diagnostics/ (per-dataset MCMC diagnostic CSVs)"
echo "  figures/mcmc/            (trace + diagnostic plots)"
echo "  figures/{geco,dundee}/    (regenerated cross-dataset plots)"
