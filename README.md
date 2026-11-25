# Fusion_ST_variants

> **Fusion Strategies and Enhancement Mechanisms in Spatio-Temporal Traffic Prediction: A Systematic Review with a Controlled Comparison of Graph and Temporal Convolutional Networks**
>
> Empirical code accompanying the manuscript.
>
> **Authors:** Omar S. Aba Hussen · Shaiful J. Hashim · Khairulmizam Samsudin · Helmi Z. M. Shafri

---

## Overview

`Fusion_ST_variants` is the controlled empirical component of a two-part study:

1. **Systematic review** — 123 peer-reviewed journal articles (2018–March 2026) synthesised into an *enhancement-oriented taxonomy* that organises spatial, temporal and fusion techniques by refinement mechanism rather than backbone.
2. **Controlled comparison** — one model family of that taxonomy (GCN–TCN) is tested in **255 valid single-seed experiments**: 6 GCN variants × 4 TCN variants, 4 parallel fusion operators and a plain stacked baseline, on METR-LA and PEMS-BAY at three horizons, with every variant in a deliberately reduced form under one training budget. One leading configuration per condition is retrained with two further seeds.

### Headline results (within this GCN–TCN scope)

- **The cost of stacking depends on the spatial encoder** — parallel fusion wins **35 of 44** configuration-matched comparisons (exact sign test p ≈ 10⁻⁴; median MAE reduction 28.1%): **24 of 24** for encoders without a shortcut (median 40.2%) and **11 of 20** for encoders with a shortcut and layer normalisation (median 0.4%). In congested traffic (< 40 mph, missing readings excluded) the two groups give opposite results (24 of 24 and 4 of 20).
- **Post-hoc shortcut control** — adding only a learned shortcut to the stacked GCN, dGCN and Spectral GCN encoders (H = 12) brings their MAE to 0.97–1.00 of plain parallel FGM; layer normalisation alone (GCN) changes it by less than 0.2%. See `posthoc_shortcut_runs.py`.
- **Fusion operator** — gated fusion (FGM) is primary in 5 of 6 conditions; attention fusion (FAM) at PEMS-BAY, H = 12.
- **Component choice** — parallel-arm MAE spreads are 0.031–0.129; component and integration effects interact and are not ranked.
- **Training instability** — the Spectral GCN (key `GWN`, a fixed sum of Laplacian powers) paired with FAM has a non-finite training loss in all 14 attempted runs under half-precision training.

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
│   ├── run_settings.csv            # Recorded setting of each of the 255 runs
│   └── evaluate_masked.py          # Inference-only re-evaluation with missing (zero-speed) readings excluded
│
├── train_ablation.py               # Core training engine — called by ALL server scripts
│
├── server_s1_fusion_screen.py      # Step 1 : Fusion screening
├── server_s2_full_sweep.py         # Step 2 : Full GCN × TCN sweep
├── server_s3_fusion_confirm.py     # Step 2b: Fusion confirmation (secondary fusions)
├── server_s3_stacked.py            # Step 3 : Stacked architecture baseline
├── server_s4_multiseed.py          # Seed-stability runs (seeds 123, 456)
├── posthoc_shortcut_runs.py        # Post-hoc runs: 14 shortcut controls + 3 single-precision reruns
├── rerun_recorded.py               # Re-train any recorded run with its recorded setting
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

## Experimental Protocol

A full-factorial design would require 6 × 4 × 5 × 2 × 3 = **720 single-seed runs**.
Instead, a *taxonomy-informed sequential-filtering strategy* is used. The manuscript describes it
as three stages (fusion screening, full GCN × TCN sweep, stacked baseline); the scripts below
implement those stages plus the fusion-confirmation runs and the seed-stability runs.

| Step | Script | Purpose |
|------|--------|---------|
| **1 — Fusion Screening** | `server_s1_fusion_screen.py` | Select the primary fusion per (dataset × horizon) on representative encoder pairs. **Must run before Step 2.** |
| **2 — Full Encoder Sweep** | `server_s2_full_sweep.py` | All 6 GCN × 4 TCN combinations with the primary fusion per condition fixed. Answers **RQ1** and **RQ2**. |
| **2b — Fusion Confirmation** | `server_s3_fusion_confirm.py` | Secondary fusion methods on a subset of encoder pairs. Used for **RQ3**. |
| **3 — Stacked Baseline** | `server_s3_stacked.py` | Encoder pairs in stacked (sequential) mode: the parallel vs. stacked comparison for **RQ3**. |
| **Seed stability** | `server_s4_multiseed.py` | One leading configuration per condition retrained with seeds 123 and 456 (coefficient of variation 0.15–0.94%). |

Run counts: **280** training runs at the primary seed (42), of which 25 anomalous runs are
excluded (see below), leaving **255 valid experiments**; plus **12** seed-stability runs.

### Step 1 results (already coded in `server_s2_full_sweep.py`)

The primary fusion per condition is the one with the lowest mean validation loss over the Step 1 grid. The original screen ranked fusions by mean test MAE; `analysis/stage1_selection_check.py` re-ranks the same runs by validation loss and selects the same fusion in all six conditions (Supplementary Table S2 of the manuscript).

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

### Required manual edits before the fusion-confirmation and seed-stability scripts

| Script | Variable | How to determine |
|--------|----------|-----------------|
| `server_s3_fusion_confirm.py` | `BEST_GCN`, `BEST_TCN` | GCN+TCN pair winning most conditions in the Step 2 CSV |
| `server_s4_multiseed.py` | `TOP_CONFIGS` | Best `(gcn, tcn, fusion)` per `(dataset, horizon)` from the Step 2 CSV |

### Anomalous runs (excluded from analysis)

| Type | Count | Description |
|------|:-----:|-------------|
| **GWN + FAM divergence** | 14 | Non-finite training loss from epoch 1 in all six conditions — a reproducible instability of this spectral-filter implementation with attention fusion |
| **GCN+AM + PEMS-BAY NaN** | 10 | NaN validation MAE from epoch 1 on PEMS-BAY (10 of 15 GCN+AM PEMS-BAY runs; the other 5 complete normally) |
| **Incomplete test evaluation** | 1 | GWN + AMs + FGM on PEMS-BAY H=12; test-set evaluation interrupted before results were recorded |

Names used in the manuscript for the keys below: `GWN` = Spectral GCN, `DiffusionGCN` = dGCN,
`GCN+AM` = GCN+AMs, `GCN+Gating` = GCN+Gate, `Default` = D-TCN, `AMs` / `EnStr` / `STech` = TCN+AMs / TCN+EnStr / TCN+STech.
D-TCN is two causal convolution layers (kernel 2, dilation 1) and its output sees the last 3 of 12 input steps; `EnStr` sees the last 4.

Three stacked Spectral GCN + TCN+AMs METR-LA runs (H = 3, 6, 12) also had a non-finite loss in every epoch; they are replaced by
single-precision reruns (`posthoc_shortcut_runs.py`, runs 8–10) in the matched-pair comparison.

---

## Components

### GCN Variants

| Key | Class | Taxonomy Category | Step 2 |
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

| Key | Taxonomy Category | Description | Step 2 |
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
# Step 1 — Fusion screening  ← run first; required before Step 2
python Fusion_ST_variants/server_s1_fusion_screen.py

# Step 2 — Full GCN × TCN sweep  ← verify BEST_FUSION_MAP in script first
python Fusion_ST_variants/server_s2_full_sweep.py

# Step 2b — Fusion confirmation  ← set BEST_GCN / BEST_TCN from Step 2 results
python Fusion_ST_variants/server_s3_fusion_confirm.py

# Step 3 — Stacked architecture baseline
python Fusion_ST_variants/server_s3_stacked.py

# Seed stability  ← set TOP_CONFIGS from Step 2 results
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

The staged server scripts use the values below. The 255 analysed runs are **not uniform**: 199 used them, and 56 used TCN hidden dimension **128** and learning rate **5 × 10⁻⁴** (one of these also batch size 256). The recorded setting of every run is in `analysis/run_settings.csv`; `rerun_recorded.py` trains any run with its recorded setting. Training uses half-precision autocasting and gradient-norm clipping at 5; evaluation is in single precision.

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

| Step | Script | RTX 3050 (6 GB) | RTX 4090 (24 GB) |
|------|--------|:---------------:|:----------------:|
| 1 | `server_s1_fusion_screen.py` | ~16–22 h | ~5–8 h |
| 2 | `server_s2_full_sweep.py` | ~46–60 h | ~16–21 h |
| 2b | `server_s3_fusion_confirm.py` | ~10–14 h | ~3–5 h |
| 3 | `server_s3_stacked.py` | ~10–14 h | ~3–5 h |
| Seed stability | `server_s4_multiseed.py` | ~7–12 h | ~2–4 h |
| **Total** | | **~89–122 h** | **~29–43 h** |

> Server scripts are pre-configured for `batch_size=128, num_workers=8` (Linux server / RTX 4090).
> Reduce `BATCH_SIZE` and `NUM_WORKERS` when running locally on a smaller GPU.

---

## Reproducibility

- Data splits are **chronological**: 70% train / 20% val / 10% test (Li et al. 2018 use 10% validation and 20% test).
- Z-score scaler is **fit on the training slice only** — no data leakage into val/test.
- Primary seed: `42`. Additional seeds for the seed-stability runs: `123`, `456`.
- Completed runs are **automatically skipped** — use `--force` to re-run.
- The 25 excluded runs (GWN+FAM divergence, GCN+AM NaN on PEMS-BAY, one incomplete evaluation) are documented in the manuscript's experimental-protocol section.

---

## Key Findings Summary

| Finding | Result |
|---------|--------|
| Parallel vs. stacked (44 configuration-matched pairs) | Parallel wins 35 of 44 (exact sign test p ≈ 10⁻⁴); median MAE reduction 28.1%; 24 of 24 without a shortcut, 11 of 20 with one |
| Missing readings excluded (masked MAE) | Parallel wins 36 of 44; in congested traffic (< 40 mph) 28 of 44, p = 0.096 — not significant |
| Model size | Parallel models have median 1.41× the parameters of their stacked counterparts |
| Stacking without a shortcut | Stacked MAE is 1.35–2.61× the parallel value in all 24 pairs |
| Primary fusion | FGM in 5 of 6 conditions; FAM at PEMS-BAY H=12 |
| Temporal component | TCN+AMs (`AMs`) has the lowest marginal MAE in all six conditions; under RMSE the METR-LA ordering changes |
| Spatial component | GCN+Gate (`GCN+Gating`) and LightGAT lead the marginals; both also contain a shortcut and layer normalisation |
| Within-family MAE spread | 0.031–0.129 |
| Training instability | GWN + FAM: non-finite training loss in all 14 attempted runs |
| Seed stability (one leading configuration per condition) | Coefficient of variation 0.15–0.94% |

---

## Citation

If you use this code in your research, please cite:

```bibtex
@unpublished{AbahussenHashim2026fusionstvar,
  title   = {Fusion Strategies and Enhancement Mechanisms in Spatio-Temporal Traffic
             Prediction: A Systematic Review with a Controlled Comparison of Graph and Temporal
             Convolutional Networks},
  author  = {Aba Hussen, Omar S. and Hashim, Shaiful J. and
             Samsudin, Khairulmizam and Shafri, Helmi Z. M.},
  year    = {2026},
  note    = {Manuscript}
}
```

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## Contributing

Contributions, bug reports, and feature requests are welcome — please open a GitHub Issue.
