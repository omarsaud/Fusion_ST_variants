"""
SERVER STAGE 1 — Fusion Method Screening
==========================================
Purpose : Identify the best fusion method per (dataset × horizon) condition.
          This result is REQUIRED before running Stage 2.

Experiments: 48
  Parallel: 2 GCN × 2 TCN × 4 fusions × 2 datasets × 3 horizons = 48
  Stack:    moved to server_s3_stacked.py (all 6 GCN variants)

Outputs:
  Fusion_ST_variants_runs/results__metr-la.csv
  Fusion_ST_variants_runs/results__pems-bay.csv

After running: Open the CSVs, filter arch=parallel, group by (dataset, horizon, fusion),
sort by MAE. The winning fusion per condition determines server_s2_full_sweep.py config.

Expected (based on previous runs, verify):
  METR-LA  H=3  → FGM
  METR-LA  H=6  → FGM  (new — verify from results)
  METR-LA  H=12 → FGM
  PEMS-BAY H=3  → FGM
  PEMS-BAY H=6  → FGM  (new — verify from results)
  PEMS-BAY H=12 → FAM

Run from the 1912 parent folder:
  python Fusion_ST_variants/server_s1_fusion_screen.py

Estimated time:
  RTX 3050 (6GB):  ~16–22 hours  (48 exp)
  RTX 4090 (24GB): ~5–8 hours    (batch_size=128, num_workers=4)
"""

from __future__ import annotations
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for p in [str(_HERE), str(_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import Fusion_ST_variants.train_ablation as _t

# ── Dataset & horizons ─────────────────────────────────────────────────────────
_t.DATASET   = "[metr-la, pems-bay]"
_t.HISTORY   = 12
_t.HORIZONS  = [3, 6, 12]

# ── 2 representative GCN × 2 representative TCN for screening ─────────────────
_t.GCN_VARIANTS   = ["GCN", "GWN"]
_t.TCN_VARIANTS   = ["Default", "AMs"]
_t.ARCHITECTURES  = ["parallel"]          # stack handled by server_s3_stacked.py
_t.FUSION_METHODS = ["fgm", "fam", "direct", "pmf"]

_t.SEEDS = [42]

# ── Optimised hyperparameters (RTX 4090 / server) ─────────────────────────────
_t.GCN_DIM       = 64      # consistent with Stage 2-4
_t.TCN_DIM       = 64     # consistent with Stage 2-4 (was 64 on local GPU)
_t.EPOCHS        = 100
_t.PATIENCE      = 15
_t.LEARNING_RATE = 1e-3    # (0.001) instead of 5e-4
_t.WEIGHT_DECAY  = 1e-5
_t.DROPOUT       = 0.2
_t.BATCH_SIZE    = 128     # RTX 4090: 24GB VRAM — was 32 on RTX 3050
_t.NUM_WORKERS   = 8       # Linux server: use parallel data loading

# ── Save predictions for Stage-1 best configs (small overhead) ────────────────
_t.LOG_VAL_METRICS         = True
_t.SLEEP_BETWEEN_RUNS_S    = 1.0

if __name__ == "__main__":
    print("=" * 80)
    print("SERVER STAGE 1: Fusion Screening  (48 experiments — parallel only)")
    print("=" * 80)
    _t.main()
