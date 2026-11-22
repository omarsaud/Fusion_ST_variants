"""
SERVER STAGE 3 — Stacked (Sequential) Baseline
===============================================
Purpose : Quantify MAE penalty of sequential GCN→TCN architecture.
          Completes Table 8 (bottom row) and provides data for Figure 24.

Experiments: 36 (all new — Stage 1 is now parallel-only)
  6 GCN × 1 TCN (Default) × stack × direct × 2 datasets × 3 horizons = 36

  GCN variants used: same 6 as Stage 2 (GL and SAGE excluded, see MANUSCRIPT_SECTION.md)
  Excluded: GL (redundant with GWN adaptive adjacency)
            SAGE (not designed for road-graph topology)

Run from the 1912 parent folder:
  python Fusion_ST_variants/server_s3_stacked.py

Estimated time:
  RTX 3050 (6GB):  ~10-14 hours  (36 exp)
  RTX 4090 (24GB): ~3-5 hours    (batch_size=128, num_workers=4)
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

_t.DATASET         = "[metr-la, pems-bay]"
_t.HISTORY         = 12
_t.HORIZONS        = [3, 6, 12]
_t.GCN_VARIANTS    = ["GCN", "DiffusionGCN", "GWN", "GCN+AM", "LightGAT", "GCN+Gating"]  # 6 variants (GL+SAGE excluded)
_t.TCN_VARIANTS    = ["Default"]
_t.ARCHITECTURES   = ["stack"]
_t.FUSION_METHODS  = ["direct"]   # stack arch only uses direct
_t.SEEDS           = [42]

_t.GCN_DIM       = 64
_t.TCN_DIM       = 64      # aligned with Stage 1/2/3a/4 (was 128)
_t.EPOCHS        = 100
_t.PATIENCE      = 15
_t.LEARNING_RATE = 1e-3    # (0.001) instead of 5e-4
_t.WEIGHT_DECAY  = 1e-5
_t.DROPOUT       = 0.2
_t.BATCH_SIZE    = 128     # RTX 4090: 24GB VRAM
_t.NUM_WORKERS   = 8       # Linux server
_t.LOG_VAL_METRICS      = True
_t.SLEEP_BETWEEN_RUNS_S = 1.0

if __name__ == "__main__":
    print("=" * 80)
    print("SERVER STAGE 3: Stacked Baseline (36 experiments — 6 GCN × 2 datasets × 3 horizons)")
    print("=" * 80)
    _t.main()
