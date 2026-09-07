# KARMA
**K-order Approximation via Markov chains for Retrospective Attribution**

KARMA is a model-agnostic framework for retrospective causal attribution in multivariate time series. It recovers a sparse lag-order DAG over the variables by building a Markov transition kernel from a pre-trained oracle, then ranking edges by their kernel TV contribution (ρ).

---

## Reproducing Paper Results

All tables and figures in the paper come from compare_realdata_cv and compare_varmulti experiment scripts. The steps below reproduce them end-to-end.

### 1 — Install dependencies

Just build devcontainer and all dependencies will be installed automatically. 
### 2 — Prepare data

Raw datasets must be placed under `data/raw/<dataset>/` before generating `.npy` splits. The pipeline expects pre-processed windows at `data/generated/<dataset>/X_train.npy` and `X_val.npy`. Datasets already pre-processed for the paper are provided under `data/generated/`. To generate your own:
```bash

### 3 — Train oracles (LSTM / TCN)

Checkpoints are already provided under `outputs/checkpoints/`. To retrain from scratch:

```bash
# Single dataset + architecture
python -m pipeline.training_pipeline --dataset exchange_rate --model lstm

# All datasets, both architectures
python -m pipeline.training_pipeline --dataset all --model both --epochs 50
```

Checkpoints are saved to the path specified in `configs/datasets/<dataset>.yaml` (e.g. `outputs/checkpoints/exchange_rate_lstm/best.pt`).

---

### 4 — Run Cross Validation Experiments for TCN (main paper results)

The KARMA pipeline runs all three pillars (discretisation → K*/b* selection → kernel estimation + DAG recovery) and writes results to `results/<dataset>/<model>/`.

**TCN Experiment in Paper for all Metrics**

```bash
python -m experiments.comparison_realdata_cv --skip_gru --skip_lstm --skip_transformer --skip_rf --datasets etth1 exchange_rate beijing_pm25 etth2 ettm1 ettm2 electricity

python -m experiments.comparison_realdata_cv --skip_gru --skip_lstm --skip_transformer --skip_rf --datasets web_traffic --tau '0,1,2'
```

**All datasets in the paper and Supplementary Material**

```bash
python -m experiments.comparison_realdata_cv
```

### 5 — Synthetic VAR experiment (Table 1)

**What this validates:**  
KARMA's causal discovery accuracy on a known ground-truth VAR(3) process. Since the true causal structure is analytical, we can directly compare KARMA's edge rankings (via Kendall's τ) to the true coefficient magnitudes. This tests:
- Whether KARMA correctly recovers lag order (K*) in synthetic settings
- Ranking quality across 5 VAR configurations (tiny → xlarge), where graph density increases
- Performance vs. 3 baselines under increasing complexity

**Expected result (Table 1):**  
KARMA should be the sole top-ranked method on medium, large, and xlarge configurations, achieving τ ≥ 0.90 on these harder cases.

```bash
# Reproduce Table 1 (default VAR(3) config)
python -m experiments.comparison_var
```

Output: `results/comparison_var/results.json` with Kendall τ scores.

**Multi-scale VAR experiment** (Appendix):

Scans KARMA across 5 VAR graph densities to show robustness as complexity increases.

```bash
# Full sweep (all configurations)
python -m experiments.comparison_var_multi --configs tiny small medium large xlarge
```

Output: `results/comparison_var_multi/` with per-config τ scores.

---

### 6 — Real-data AUC comparison (Tables 2–3 & Figure 4)

**What this validates:**  
KARMA's faithfulness on real-world time series forecasting tasks under a deletion-based faithfulness protocol (AUC_lag). Unlike synthetic data, we cannot know the true causal graph, but we measure how well each method's top-ranked lags degrade prediction when occluded:
- **Lag-AUC (Table 2):** Higher is better. Measures ranking quality: removing top-ρ lags should hurt prediction most
- **Complexity (Table 3):** Lower is better. Measures explanation conciseness (magnitude of attribution mass)

KARMA also reports reliability certificates (Level 5) for each explanation, unavailable from baselines.

**Expected result (Tables 2–3):**  
- Lag-AUC: KARMA outperforms or ties baselines on 6/7 datasets (ETTh1, ETTh2, ETTm1, ExRA, Bei, Elec), with particularly strong gains on high-D dataset ExRA (D=100)
- Complexity: KARMA consistently lowest (< 2.6 across all datasets), vs. baselines ≥ 2.5–5.9

**Run with 5-fold cross-validation (produces Tables 2–3):**  

```bash
# TCN model only (matches paper Table 2–3)
python -m experiments.comparison_realdata_cv --skip_gru --skip_lstm --skip_transformer --skip_rf
```

**Run single-pass on all datasets (generates Figure 4 data):**  

```bash
# All datasets, all baselines (slower)
python -m experiments.comparison_realdata

# Specific datasets only (for quick validation)
python -m experiments.comparison_realdata --datasets etth1 exchange_rate beijing_pm25
```

Output: `results/realdata/<dataset>_results.json` with per-dataset AUC and complexity scores.

---

### 7 — Generate Figure 4 (Lag-AUC removal curves)

**Prerequisites:** Must run `comparison_realdata` first (step 6).

**What this shows:**  
For each of 7 datasets, a curve showing how prediction error changes as the top-ρ ranked lags are progressively occluded. Higher curve = method ranked important lags better.



---

## Project layout

```
configs/
  datasets/       # per-dataset YAML (paths, var names, window size)
  experiments/    # karma.yaml  — default hyperparameters
  models/         # lstm.yaml, tcn.yaml

pipeline/
  training_pipeline.py   # train LSTM/TCN oracles
  karma_pipeline.py      # full KARMA run (entry point)
  timeshap_pipeline.py   # TimeShap baseline wrapper
  visualization.py       # DAG heatmaps, convergence plots

experiments/
  comparison_var.py          # Synthetic VAR(3) vs baselines
  comparison_var_multi.py    # Multi-scale VAR sweep
  comparison_realdata.py     # Real-data AUC comparison
  plot_lag_auc_curves.py     # Figure 4 plotting script

karma/
  markov_approximation/      # Pillar 2: K* / b* selection
  causal_recovery/           # Pillar 3: edge contributions
  utils/                     # Discretiser, SuffixPool, BStarKernelEstimator

outputs/checkpoints/         # Pre-trained oracle checkpoints (best.pt)
results/                     # All experiment outputs
data/generated/              # Pre-processed .npy splits
```

---

## Quick reference: paper → script mapping

| Paper element | Script / command |
|---|---|
| Table 1 — VAR Experiment | `python -m experiments.comparison_var_multi` ||
| Tables 2 & 3 — Lag-AUC removal curves and complexity | `python -m experiments.comparison_realdata_cv --skip_gru --skip_lstm --skip_transformer --skip_rf` |
