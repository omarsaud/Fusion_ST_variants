"""
SERVER STAGE 3 — Fusion Method Confirmation (RQ3)
==================================================
Purpose : Directly answers RQ3:
          "Which parallel fusion method delivers the best accuracy?"

Strategy: Uses the single best-performing GCN and TCN pair found in Stage 2
          and sweeps ALL 4 fusion methods across every condition.
          This is more scientifically rigorous than Stage 1 (which used
          proxy models) because it uses the true optimal encoder pair.

IMPORTANT — USER MUST FILL IN BEST_GCN and BEST_TCN BEFORE RUNNING:
  After Stage 2 is complete, identify the GCN and TCN variant that wins
  most frequently across conditions. Then set BEST_GCN and BEST_TCN below.

Experiments: 30 total
  Parallel: 1 GCN × 1 TCN × 4 fusions × 2 datasets × 3 horizons = 24
  Stack:    1 GCN × 1 TCN × 1 fusion (direct) × 2 datasets × 3 horizons = 6
  Total = 24 + 6 = 30 experiments

  The stack run anchors the parallel vs. stacked comparison (RQ3 baseline).

Run from the 1912 parent folder:
  python Fusion_ST_variants/server_s3_fusion_confirm.py

Estimated time:
  RTX 3050 (6GB):  ~10-14 hours  (30 exp)
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

# ─── USER MUST CONFIGURE AFTER STAGE 2 ───────────────────────────────────────
#
# Set these to the GCN and TCN that won most frequently in Stage 2.
# Example: if LightGAT + AMs won in 4 out of 6 conditions, use those.
#
# Tie-breaking rule: prefer the variant with lower average MAE across all
# conditions where it tied for first place.
# ─────────────────────────────────────────────────────────────────────────────
BEST_GCN = "LightGAT"   # <-- Replace with your Stage 2 winner
BEST_TCN = "AMs"        # <-- Replace with your Stage 2 winner

# ─── STAGE 3 CONFIGURATION ───────────────────────────────────────────────────
_t.DATASET        = "[metr-la, pems-bay]"
_t.HISTORY        = 12
_t.HORIZONS       = [3, 6, 12]
_t.GCN_VARIANTS   = [BEST_GCN]
_t.TCN_VARIANTS   = [BEST_TCN]
_t.ARCHITECTURES  = ["parallel", "stack"]
_t.FUSION_METHODS = ["fgm", "fam", "pmf", "direct"]  # stack uses only "direct" automatically
_t.SEEDS          = [42]

_t.GCN_DIM       = 64
_t.TCN_DIM       = 64
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
    n_parallel = 1 * 1 * len(_t.FUSION_METHODS) * 2 * len(_t.HORIZONS)   # 24
    n_stack    = 1 * 1 * 1 * 2 * len(_t.HORIZONS)                         # 6
    n = n_parallel + n_stack
    print("=" * 80)
    print(f"SERVER STAGE 3: Fusion Method Confirmation ({n} experiments: {n_parallel} parallel + {n_stack} stack)")
    print(f"  Best GCN : {BEST_GCN}")
    print(f"  Best TCN : {BEST_TCN}")
    print(f"  Fusions  : {_t.FUSION_METHODS}")
    print(f"  Horizons : {_t.HORIZONS}")
    print(f"  Answers  : RQ3 — which fusion method is best?")
    print("=" * 80)
    _t.main()
