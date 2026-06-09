# Fusion_ST_variants

> **Spatio-Temporal Traffic Forecasting via Ablation Study of GCN and TCN Variants**
>
> Research codebase accompanying the paper *"Enhancement Mechanisms and Fusion
> Strategies in Spatio-Temporal Traffic Prediction: A Taxonomy Validated by 267
> Controlled Experiments"* (submitted to *Information Fusion*, Elsevier).
>
> This codebase produced the **267 controlled experiments** reported in the paper,
> run as a **four-stage protocol** on **METR-LA** and **PEMS-BAY**:
> **Stage 1** fusion screening (`server_s1_fusion_screen.py`),
> **Stage 2** full GCN×TCN sweep (`server_s2_full_sweep.py`),
> **Stage 3** stacked/sequential baseline (`server_s3_stacked.py`, `server_s3_fusion_confirm.py`),
> and **Stage 4** multi-seed validation (`server_s4_multiseed.py`).

---

## Overview

`Fusion_ST_variants` is a modular, ablation-ready framework for evaluating combinations of **Graph Convolutional Network (GCN)** and **Temporal Convolutional Network (TCN)** variants for traffic-speed forecasting on standard benchmark datasets (METR-LA, PEMS-BAY).

The framework supports:

- **8 GCN spatial encoders** — from vanilla GCN to attention-augmented variants
- **4 TCN temporal encoders** — default, enhanced strided, gated multi-scale, and adaptive multi-scale
- **2 coupling architectures** — `stack` (GCN → TCN in series) and `parallel` (GCN ‖ TCN with fusion head)
- **4 fusion methods** (parallel arch only) — `direct`, `fgm`, `pmf`, `fam`
- Automatic caching of preprocessed data (Z-score normalized, per-split scaler)
- Reproducible multi-seed ablation sweeps with CSV result logging

---

## Repository Structure

```
Fusion_ST_variants/
├── models/
│   ├── forecaster.py        # GcnTcnForecaster + ForecasterConfig
│   ├── gcn_variants.py      # 8 GCN spatial encoder modules
│   └── tcn_variants.py      # 4 TCN temporal encoder modules
├── preprocessing/
│   ├── cache.py             # Caching of preprocessed speed & adjacency matrices
│   ├── load_raw.py          # Raw dataset loader (h5 / npy / csv)
│   ├── scaler.py            # Z-score scaler with per-split fitting
│   ├── splits.py            # Chronological train/val/test index generation
│   └── window_dataset.py    # PyTorch Dataset for sliding-window samples
├── utils/
│   ├── io.py                # File I/O helpers (CSV append, JSON save)
│   ├── metrics.py           # MAE / RMSE / MAPE / sMAPE with de-normalization
│   └── seed.py              # Reproducibility utilities
├── train_ablation.py        # Main entry point for ablation runs
├── requirements.txt
└── README.md
```

---

## GCN Variants

| Key              | Class                          | Description                                    |
| ---------------- | ------------------------------ | ---------------------------------------------- |
| `GCN`          | `GCNLayer`                   | Vanilla graph convolution:$\hat{A}X W$       |
| `DiffusionGCN` | `DiffusionGCN`               | K-hop diffusion aggregation                    |
| `GL`           | `GraphAdaptiveLearning`      | Adaptive adjacency from feature similarity     |
| `SAGE`         | `GraphSAGE`                  | Neighbor aggregation + self-concat (GraphSAGE) |
| `GWN`          | `GraphWaveletNetwork`        | Chebyshev-approximated wavelet transform       |
| `GCN+AM`       | `GCNWithAttention`           | GCN + multi-head attention masked by adjacency |
| `GCN+Gating`   | `GCNWithGating`              | GCN with feature and update gates + residual   |
| `LightGAT`     | `LightGraphAttentionNetwork` | Lightweight attention with adjacency masking   |

## TCN Variants

| Key         | Description                 |
| ----------- | --------------------------- |
| `Default` | Standard dilated causal TCN |
| `EnStr`   | Enhanced strided TCN        |
| `GMs`     | Gated multi-scale TCN       |
| `AMs`     | Adaptive multi-scale TCN    |

## Fusion Methods (parallel architecture only)

| Key        | Description                                                                    |
| ---------- | ------------------------------------------------------------------------------ |
| `direct` | Concatenation only (baseline)                                                  |
| `fgm`    | Feature Gating Mechanism — sigmoid gate on concatenated features              |
| `pmf`    | Parametric Matrix Fusion — learnable linear mixing matrix                     |
| `fam`    | Fusion Attention Mechanism — multi-head self-attention over combined features |

---

## Datasets

| Dataset  | Nodes | Timesteps | Freq  | Source                                   |
| -------- | ----- | --------- | ----- | ---------------------------------------- |
| METR-LA  | 207   | 34,272    | 5-min | [DCRNN](https://github.com/liyaguang/DCRNN) |
| PEMS-BAY | 325   | 52,116    | 5-min | [DCRNN](https://github.com/liyaguang/DCRNN) |

Place raw dataset files under a `data/` directory at the **project root** (the parent directory of the `Fusion_ST_variants/` package):

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

# 2. Create and activate a virtual environment (recommended)
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

Requires **Python ≥ 3.9** and **PyTorch ≥ 2.0** (CUDA optional but recommended).

---

## Usage

### Running an ablation sweep

Edit the top-level constants in `train_ablation.py` (or pass CLI flags):

```python
DATASET        = "[pems-bay]"                       # dataset(s) to run
HORIZONS       = [3, 6, 12]                         # prediction horizons (× 5-min steps)
GCN_VARIANTS   = ["GCN", "GWN", "GCN+AM"]
TCN_VARIANTS   = ["Default", "AMs"]
ARCHITECTURES  = ["parallel", "stack"]
FUSION_METHODS = ["fgm", "direct", "pmf", "fam"]   # parallel arch only
SEEDS          = [42]
```

Then run from the **project root** (parent of the package folder):

```bash
# Single dataset
python -m Fusion_ST_variants.train_ablation --dataset pems-bay

# Both datasets
python -m Fusion_ST_variants.train_ablation --dataset "[metr-la, pems-bay]"

# Quick smoke test (1 batch per phase)
python -m Fusion_ST_variants.train_ablation --smoke

# Override variants via CLI
python -m Fusion_ST_variants.train_ablation \
    --dataset metr-la \
    --gcn_variants "GWN,GCN+AM" \
    --tcn_variants "AMs" \
    --architectures parallel \
    --fusion_methods "fgm,fam" \
    --horizons "3,6,12" \
    --epochs 50
```

### Key CLI flags

| Flag                 | Default                | Description                                |
| -------------------- | ---------------------- | ------------------------------------------ |
| `--dataset`        | `pems-bay`           | Dataset key(s):`metr-la` or `pems-bay` |
| `--horizons`       | `3`                  | Comma-separated list of horizon steps      |
| `--gcn_variants`   | `GWN`                | GCN variant keys                           |
| `--tcn_variants`   | `AMs`                | TCN variant keys                           |
| `--architectures`  | `parallel,stack`     | `parallel` and/or `stack`              |
| `--fusion_methods` | `fgm,direct,pmf,fam` | Fusion method keys                         |
| `--epochs`         | 40                     | Max training epochs                        |
| `--patience`       | 10                     | Early stopping patience                    |
| `--batch_size`     | 64                     | Mini-batch size                            |
| `--lr`             | `1e-3`               | Learning rate                              |
| `--dropout`        | `0.2`                | Dropout rate                               |
| `--gcn_dim`        | 32                     | GCN hidden dimension                       |
| `--tcn_dim`        | 64                     | TCN hidden dimension                       |
| `--force`          | —                     | Re-run even if `metrics.json` exists     |
| `--smoke`          | —                     | Quick sanity check (10 batches per phase)  |
| `--save_preds`     | —                     | Save `preds_test.npz` for each run       |
| `--no_amp`         | —                     | Disable automatic mixed precision          |
| `--device`         | `cuda`               | PyTorch device (`cuda` or `cpu`)       |

### Outputs

Each run is saved to `<project_root>/Fusion_ST_variants_runs/<run_name>/`:

```
Fusion_ST_variants_runs/
└── pems-bay__h03__GWN__AMs__parallel__fgm__seed42__psm_all_meanpool/
    ├── config.json          # full hyperparameter snapshot
    ├── best.pt              # best model checkpoint
    ├── metrics.json         # final test metrics
    └── epoch_history.json   # per-epoch train/val loss log
```

A consolidated CSV is written to:

```
Fusion_ST_variants_runs/results__<dataset>.csv
```

---

## Metrics

All metrics are computed **after inverse Z-score scaling** (original speed units):

| Metric | Description                                  |
| ------ | -------------------------------------------- |
| MAE    | Mean Absolute Error                          |
| RMSE   | Root Mean Squared Error                      |
| MAPE   | Mean Absolute Percentage Error (zero-masked) |
| sMAPE  | Symmetric MAPE                               |

Horizon-specific metrics (`@3`, `@6`, `@12`) correspond to 15-min, 30-min, and 60-min predictions.

---

## Reproducibility

- All experiments use a fixed random seed (`--seeds 42` by default).
- Data splits are chronological: 70% train / 20% val / 10% test (by time index).
- The Z-score scaler is fit on the **training slice only** and applied to val/test.
- Completed runs are automatically skipped (use `--force` to override).

---

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{OMAR2026fusionstvar,
  title   = {Enhancement Mechanisms and Fusion Strategies in Spatio-Temporal Traffic Prediction: A Taxonomy Validated by 267 Controlled Experiments},
  author  = {Omar Saud Abahussen},
  journal = {Information Fusion},
  year    = {2026}
}
```

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## Contributing

Contributions, bug reports, and feature requests are welcome — please open a GitHub Issue.
