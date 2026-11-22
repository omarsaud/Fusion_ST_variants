# Fusion_ST_variants

> **Enhancement Mechanisms and Fusion Strategies in Spatio-Temporal Deep Learning for Traffic Prediction: A Systematic Review and Empirical Analysis**
>
> Empirical codebase accompanying a peer-reviewed manuscript (under review).
>
> **Authors:** Omar S. Aba Hussen · Shaiful J. Hashim · Khairulmizam Samsudin · Helmi Z. M. Shafri
>
> **GitHub:** https://github.com/omarsaud/Fusion_ST_variants *(private repository — will be made public upon manuscript acceptance)*

---

## Overview

`Fusion_ST_variants` is the controlled empirical component of a two-stage research contribution:

1. **Systematic review** — 131 peer-reviewed articles (2018–March 2026) synthesised into an *enhancement-oriented taxonomy* classifying GCN, TCN, and fusion techniques by their underlying refinement mechanisms.
2. **Empirical validation** — The taxonomy is stress-tested through **267 valid experiments** spanning 6 GCN variants × 4 TCN variants × 5 fusion methods × 2 benchmark datasets × 3 prediction horizons, with 6 top configurations re-validated across 3 random seeds.

### Headline results

- **Spatial enhancement is the dominant design axis** — GCN variant selection produces **1.15–1.98× larger MAE spread** than TCN selection across all six dataset–horizon conditions.
- **Parallel integration consistently outperforms sequential (stacked) designs** — **24.5–49.8% MAE reduction** over stacked baselines; largest advantage at short horizons where sequential spatial pre-smoothing creates an information bottleneck.
- **Fusion method is condition-sensitive** — FGM (gated) is optimal in **5 of 6 conditions**; FAM (attention) wins for PEMS-BAY H=12 (dense network, long horizon). An architectural incompatibility is reproducibly characterised: **GWN + FAM → gradient explosion** across all 14 pairings.

---

## Research Questions

| RQ | Question |
|----|----------|
| **RQ1** | How do different spatial (GCN) enhancement mechanisms affect prediction accuracy across network topologies and prediction horizons? |
| **RQ2** | How do different temporal (TCN) enhancement mechanisms affect accuracy, and does their relative benefit depend on the horizon or dataset? |
| **RQ3** | How do parallel fusion strategies compare with stacked (sequential) integration, and does the optimal fusion subtype vary across dataset–horizon conditions? |

---

## Repository Structure

```
Fusion_ST_variants/
├── models/
│   ├── forecaster.py               # GcnTcnForecaster + ForecasterConfig
│   ├── gcn_variants.py             # 8 GCN spatial encoder modules (6 active, 2 excluded)
│   └── tcn_variants.py             # 5 TCN temporal encoder modules (4 active, 1 excluded)
├── preprocessing/
│   ├── cache.py                    # Caching of preprocessed speed & adjacency matrices
│   ├── load_raw.py                 # Raw dataset loader (h5 / npy / csv)
│   ├── scaler.py                   # Z-score scaler with per-split fitting
│   ├── splits.py                   # Chronological train/val/test index generation
│   └── window_dataset.py           # PyTorch Dataset for sliding-window samples
├── utils/
│   ├── io.py                       # File I/O helpers (CSV append, JSON save)
│   ├── metrics.py                  # MAE / RMSE / MAPE / sMAPE with de-normalization
│   └── seed.py                     # Reproducibility utilities
├── analysis/
│   ├── diebold_mariano.py          # Diebold–Mariano test between two trained runs
│   └── evaluate_masked.py          # Inference-only re-evaluation with missing (zero-speed) readings excluded
│
├── train_ablation.py               # Core training engine — called by ALL server scripts
│
├── server_s1_fusion_screen.py      # Stage 1 : Fusion screening          (48 experiments)
├── server_s2_full_sweep.py         # Stage 2 : Full GCN × TCN sweep      (144 experiments)
├── server_s3_fusion_confirm.py     # Stage 3a: Fusion confirmation        (30 experiments)
├── server_s3_stacked.py            # Stage 3b: Stacked architecture baseline (36 experiments)
├── server_s4_multiseed.py          # Stage 4 : Multi-seed robustness      (12 new experiments)
├── server_analyze.py               # Post-run analysis: tables, figures, statistics
│
├── requirements.txt
└── README.md
```

> **`train_ablation.py` is the core training engine.** All five server scripts import it and
> override its module-level variables before calling `_t.main()`. It is **required** and must not
> be removed. Users should run the server scripts for staged experiments and `train_ablation.py`
> directly only for custom / ad-hoc runs.

---

## Four-Stage Experimental Protocol

A naïve full-factorial design would require **2,160 training runs**
(6 GCN × 4 TCN × 5 fusion × 2 datasets × 3 horizons × 3 seeds).
Instead, a *taxonomy-informed sequential filtering strategy* reduces this to 292 runs while
preserving complete coverage of the most informative region of the design space.

| Stage | Script | Experiments | Purpose |
|-------|--------|:-----------:|---------|
| **1 — Fusion Screening** | `server_s1_fusion_screen.py` | 48 | Identify best fusion per (dataset × horizon) using 2 representative GCN × 2 TCN proxy pairs. **Must run before Stage 2.** |
| **2 — Full Encoder Sweep** | `server_s2_full_sweep.py` | 144 | All 6 GCN × 4 TCN combinations with per-condition best fusion fixed (2 passes). Answers **RQ1** and **RQ2**. |
| **3a — Fusion Confirmation** | `server_s3_fusion_confirm.py` | 30 | All 4 fusion methods on the single best encoder pair (from Stage 2). Answers **RQ3** rigorously. |
| **3b — Stacked Baseline** | `server_s3_stacked.py` | 36 | 6 GCN × Default TCN in stacked mode. Provides the RQ3 counterfactual: parallel vs. sequential. |
| **4 — Multi-Seed Validation** | `server_s4_multiseed.py` | 12 new | Best config per condition run with seeds 123 and 456 (seed 42 already exists from Stage 2). Confirms CV < 0.93%. |
| **Total** | | **~292** | 267 valid; 25 excluded (see anomaly notes below) |

### Stage 1 results (already coded in `server_s2_full_sweep.py`)

```python
BEST_FUSION_MAP = {
    ("metr-la",   3): "fgm",   # FGM confirmed
    ("metr-la",   6): "fgm",   # FGM confirmed
    ("metr-la",  12): "fgm",   # FGM confirmed
    ("pems-bay",  3): "fgm",   # FGM confirmed
    ("pems-bay",  6): "fgm",   # FGM confirmed
    ("pems-bay", 12): "fam",   # FAM confirmed
}
```

### Required manual edits before Stage 3 and Stage 4

| Script | Variable | How to determine |
|--------|----------|-----------------|
| `server_s3_fusion_confirm.py` | `BEST_GCN`, `BEST_TCN` | GCN+TCN pair winning most conditions in Stage 2 CSV |
| `server_s4_multiseed.py` | `TOP_CONFIGS` | Best `(gcn, tcn, fusion)` per `(dataset, horizon)` from Stage 2 CSV |

### Anomalous runs (excluded from analysis)

| Type | Count | Description |
|------|:-----:|-------------|
| **GWN + FAM gradient explosion** | 14 | NaN training loss from epoch 1 across all conditions — reproducible architectural incompatibility |
| **GCN+AM + PEMS-BAY NaN** | 10 | NaN validation MAE from epoch 1 on PEMS-BAY; numerical singularity in denormalisation pass |
| **Incomplete test evaluation** | 1 | GWN + AMs + FGM on PEMS-BAY H=12; server interruption during final test step |

---

## Components

### GCN Variants

| Key | Class | Taxonomy Category | Stage 2 |
|-----|-------|-------------------|:-------:|
| `GCN` | `GCNLayer` | Backbone baseline | ✓ |
| `DiffusionGCN` | `DiffusionGCN` | Structural refinement (multi-hop diffusion) | ✓ |
| `GWN` | `GraphWaveletNetwork` | Structural refinement (adaptive adjacency, multi-scale spectral) | ✓ |
| `GCN+AM` | `GCNWithAttention` | Attention-based refinement (post-aggregation, multi-head) | ✓ |
| `GCN+Gating` | `GCNWithGating` | Gated / auxiliary mechanism (feature + update gates) | ✓ |
| `LightGAT` | `LightGraphAttentionNetwork` | Attention-based refinement (integrated, within-aggregation) | ✓ |
| `GL` | `GraphAdaptiveLearning` | *(Excluded — redundant with GWN adaptive adjacency)* | — |
| `SAGE` | `GraphSAGE` | *(Excluded — not designed for road-graph topology)* | — |

### TCN Variants

| Key | Taxonomy Category | Description | Stage 2 |
|-----|-------------------|-------------|:-------:|
| `Default` | Backbone baseline | Standard dilated causal TCN | ✓ |
| `EnStr` | Structural refinement | Multi-scale dilated / enhanced strided TCN | ✓ |
| `AMs` | Attention-based refinement | Adaptive multi-scale TCN with selective temporal attention | ✓ |
| `STech` | Spatial–temporal enrichment | TCN conditioned on spatial encoder features | ✓ |
| `GMs` | *(Excluded — sigmoid gating is a subset of AMs attention mechanism)* | Gated multi-scale TCN | — |

### Fusion Methods

| Key | Architecture | Taxonomy Category | Description |
|-----|-------------|-------------------|-------------|
| `fgm` | parallel | Sigmoid-gated | Feature Gating Mechanism — sigmoid gate on concatenated features |
| `fam` | parallel | Cross-modal attention | Fusion Attention Mechanism — multi-head self-attention over combined features |
| `pmf` | parallel | Parameter-matrix projection | Parametric Matrix Fusion — learnable linear mixing matrix |
| `direct` | parallel / stack | Baseline concatenation | Direct additive / concatenation — no learnable fusion gate |

> **Stack architecture** always uses `direct` (GCN → TCN in series; output passed through, no fusion head).

---

## Datasets

| Dataset | Nodes | Timesteps | Freq | Source |
|---------|:-----:|:---------:|:----:|--------|
| METR-LA | 207 | 34,272 | 5-min | [Li et al. 2018 — DCRNN](https://github.com/liyaguang/DCRNN) |
| PEMS-BAY | 325 | 52,116 | 5-min | [Li et al. 2018 — DCRNN](https://github.com/liyaguang/DCRNN) |

Place raw files under `data/` at the **project root** (parent directory of this repository):

```
<project_root>/
├── data/
│   ├── metr-la/
│   │   ├── metr-la.h5          # speed matrix
│   │   └── adj_mx.pkl          # adjacency matrix
│   └── pems-bay/
│       ├── pems-bay.h5
│       └── adj_mx_bay.pkl
└── Fusion_ST_variants/         # this repository
```

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/omarsaud/Fusion_ST_variants.git
cd Fusion_ST_variants
```

**Option A — Conda (recommended for GPU / CUDA environments):**

```bash
conda create -n fusion_st python=3.9 -y
conda activate fusion_st
pip install -r requirements.txt
```

**Option B — venv:**

```bash
python -m venv venv
# Windows:  venv\Scripts\activate
# Linux:    source venv/bin/activate
pip install -r requirements.txt
```

Requires **Python ≥ 3.9** and **PyTorch ≥ 2.0** (CUDA strongly recommended for all staged runs).

---

## Usage

### Reproducing the paper experiments (staged server scripts)

Run all commands from the **project root** (the parent directory of `Fusion_ST_variants/`):

```bash
# Stage 1 — Fusion screening  ← run first; required before Stage 2
python Fusion_ST_variants/server_s1_fusion_screen.py

# Stage 2 — Full GCN × TCN sweep  ← verify BEST_FUSION_MAP in script first
python Fusion_ST_variants/server_s2_full_sweep.py

# Stage 3a — Fusion confirmation  ← set BEST_GCN / BEST_TCN from Stage 2 results
python Fusion_ST_variants/server_s3_fusion_confirm.py

# Stage 3b — Stacked architecture baseline
python Fusion_ST_variants/server_s3_stacked.py

# Stage 4 — Multi-seed robustness  ← set TOP_CONFIGS from Stage 2 results
python Fusion_ST_variants/server_s4_multiseed.py

# Post-run analysis: generate tables and figures
python Fusion_ST_variants/server_analyze.py
```

### Custom / ad-hoc runs via `train_ablation.py`

`train_ablation.py` can also be called directly via CLI for exploratory single runs:

```bash
# Quick smoke test — verifies environment and data loading (10 batches per phase)
python -m Fusion_ST_variants.train_ablation --smoke

# Single configuration
python -m Fusion_ST_variants.train_ablation \
    --dataset metr-la \
    --gcn_variants "LightGAT" \
    --tcn_variants "AMs" \
    --architectures parallel \
    --fusion_methods "fgm" \
    --horizons "3,6,12" \
    --epochs 100

# Both datasets, representative subset
python -m Fusion_ST_variants.train_ablation \
    --dataset "[metr-la, pems-bay]" \
    --gcn_variants "GCN,GWN,LightGAT" \
    --tcn_variants "Default,AMs" \
    --architectures "parallel,stack" \
    --fusion_methods "fgm,direct"
```

### Key CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | `pems-bay` | `metr-la`, `pems-bay`, or `[metr-la, pems-bay]` |
| `--horizons` | `3` | Comma-separated horizon steps (3 = 15 min, 6 = 30 min, 12 = 60 min) |
| `--gcn_variants` | `GWN` | GCN variant keys (comma-separated) |
| `--tcn_variants` | `AMs` | TCN variant keys (comma-separated) |
| `--architectures` | `parallel,stack` | `parallel` and/or `stack` |
| `--fusion_methods` | `fgm,direct,pmf,fam` | Fusion keys (stack always uses `direct` regardless) |
| `--epochs` | 40 | Max training epochs |
| `--patience` | 10 | Early stopping patience |
| `--batch_size` | 64 | Mini-batch size |
| `--lr` | `1e-3` | Learning rate |
| `--dropout` | `0.2` | Dropout rate |
| `--gcn_dim` | 32 | GCN hidden dimension |
| `--tcn_dim` | 64 | TCN hidden dimension |
| `--force` | — | Re-run even if `metrics.json` already exists |
| `--smoke` | — | Quick sanity check (10 batches per phase) |
| `--save_preds` | — | Save `preds_test.npz` for each run |
| `--no_amp` | — | Disable automatic mixed precision (AMP) |
| `--device` | `cuda` | PyTorch device (`cuda` or `cpu`) |

---

## Outputs

Each run is saved to `<project_root>/Fusion_ST_variants_runs/<run_name>/`:

```
Fusion_ST_variants_runs/
└── metr-la__h03__LightGAT__AMs__parallel__fgm__seed42__psm_all_meanpool/
    ├── config.json          # full hyperparameter snapshot
    ├── best.pt              # best model checkpoint (min val loss)
    ├── metrics.json         # final test metrics (denormalised)
    ├── epoch_history.json   # per-epoch train/val loss and val metrics
    ├── node_metrics.npz     # per-node MAE/RMSE arrays (always saved)
    └── preds_test.npz       # y_true/y_pred arrays (only if --save_preds)
```

Consolidated CSV per dataset:

```
Fusion_ST_variants_runs/
├── results__metr-la.csv
└── results__pems-bay.csv
```

---

## Metrics

All metrics are computed **after inverse Z-score scaling** (original speed units, mph):

| Metric | Description |
|--------|-------------|
| MAE | Mean Absolute Error — primary ranking criterion and optimisation target |
| RMSE | Root Mean Squared Error — penalises large errors more strongly |
| MAPE | Mean Absolute Percentage Error (zero-speed-masked; reported with support %) |
| sMAPE | Symmetric MAPE |

Horizon-specific metrics (`@3`, `@6`, `@12`) correspond to **15-min**, **30-min**, and **60-min** predictions.

### Masked re-evaluation

MAE and RMSE above are averaged over all test readings, including the zero-speed
readings that METR-LA and PEMS-BAY use for missing data. To evaluate trained runs
with those readings excluded, reload the stored checkpoints (no training):

```bash
python -m Fusion_ST_variants.analysis.evaluate_masked --out masked_eval.csv
```

This writes, per run, masked MAE/RMSE over all steps and at the final step Q,
masked MAE on congested (< 40 mph) and free-flow readings, and a check that the
recomputed unmasked MAE equals the value in `metrics.json`.

---

## Hyperparameter Configuration

The following values are applied uniformly across all staged experiments (Stages 1–4), consistent with the values reported in the manuscript:

| Hyperparameter | Value |
|----------------|-------|
| GCN hidden dimension | 64 |
| TCN hidden dimension | 64 |
| Dropout | 0.2 |
| Optimiser | Adam |
| Learning rate | 1 × 10⁻³ |
| Weight decay | 1 × 10⁻⁵ |
| Batch size | 128 |
| Max training epochs | 100 |
| Early-stop patience | 15 epochs |
| Input history window | 12 steps (60 min) |
| Input feature | Speed only |
| Framework | PyTorch ≥ 2.0 |

> The `train_ablation.py` defaults differ (batch_size=64, epochs=40, patience=10, gcn_dim=32) and are intended for local/exploratory runs. The server scripts override these values to match the paper configuration.

---

## Runtime Estimates

| Stage | Script | Experiments | RTX 3050 (6 GB) | RTX 4090 (24 GB) |
|-------|--------|:-----------:|:---------------:|:----------------:|
| 1 | `server_s1_fusion_screen.py` | 48 | ~16–22 h | ~5–8 h |
| 2 | `server_s2_full_sweep.py` | 144 | ~46–60 h | ~16–21 h |
| 3a | `server_s3_fusion_confirm.py` | 30 | ~10–14 h | ~3–5 h |
| 3b | `server_s3_stacked.py` | 36 | ~10–14 h | ~3–5 h |
| 4 | `server_s4_multiseed.py` | 12 | ~7–12 h | ~2–4 h |
| **Total** | | **~270** | **~89–122 h** | **~29–43 h** |

> Server scripts are pre-configured for `batch_size=128, num_workers=8` (Linux server / RTX 4090).
> Reduce `BATCH_SIZE` and `NUM_WORKERS` when running locally on a smaller GPU.

---

## Reproducibility

- Data splits are **chronological**: 70% train / 20% val / 10% test (DCRNN protocol, Li et al. 2018).
- Z-score scaler is **fit on the training slice only** — no data leakage into val/test.
- Primary seed: `42`. Additional seeds for Stage 4: `123`, `456`.
- Completed runs are **automatically skipped** — use `--force` to re-run.
- All 25 anomalous runs (GWN+FAM divergence, GCN+AM NaN on PEMS-BAY) are reproducible and documented in the manuscript (Section 5.1.3).

---

## Key Findings Summary

| Finding | Result |
|---------|--------|
| Spatial vs. temporal MAE spread ratio | 1.15–1.98× across all 6 conditions (spatial always dominant) |
| Best spatial variant (METR-LA) | GCN+Gating — gating suppresses noisy neighbours in sparse highway graph |
| Best spatial variant (PEMS-BAY) | LightGAT / GCN+AM — within-aggregation attention suits denser topology |
| Parallel vs. stacked MAE reduction | 24.5–49.8% (largest advantage at H=3) |
| Best fusion (5/6 conditions) | FGM (gated) |
| Best fusion (PEMS-BAY H=12) | FAM (cross-modal attention) |
| Architectural incompatibility | GWN + FAM → gradient explosion (14 runs, all datasets/horizons) |
| Multi-seed coefficient of variation | < 0.93% — confirms ordinal stability of rankings |

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{AbahussenHashim2026fusionstvar,
  title   = {Enhancement Mechanisms and Fusion Strategies in Spatio-Temporal Deep
             Learning for Traffic Prediction: A Systematic Review and Empirical Analysis},
  author  = {Aba Hussen, Omar S. and Hashim, Shaiful J. and
             Samsudin, Khairulmizam and Shafri, Helmi Z. M.},
  year    = {2026},
  note    = {Under review}
}
```

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## Contributing

Contributions, bug reports, and feature requests are welcome — please open a GitHub Issue.
