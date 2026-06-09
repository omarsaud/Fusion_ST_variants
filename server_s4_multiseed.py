"""
SERVER STAGE 4 — Multi-Seed Statistical Validation
====================================================
Purpose : Validate that the top-performing configurations from Stage 2
          produce consistent results across multiple random seeds.
          Provides data for Table 12 (Statistical Robustness).

IMPORTANT — USER MUST FILL IN TOP_CONFIGS BEFORE RUNNING:
  After Stage 2 is complete, inspect the CSV results and identify
  the best GCN+TCN combination per condition (dataset × horizon).
  Update TOP_CONFIGS below with those findings, then run this script.

Experiments: ~12 new runs
  Top-1 config per condition × 2 extra seeds (123, 456) × 6 conditions = 12
  (seed=42 already exists from Stage 2, so only seeds 123 and 456 are NEW)

Run from the 1912 parent folder:
  python Fusion_ST_variants/server_s4_multiseed.py

Estimated time:
  RTX 3050 (6GB):  ~7-12 hours  (12 exp)
  RTX 4090 (24GB): ~2-4 hours   (batch_size=128, num_workers=4)
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

# ─── USER MUST CONFIGURE THIS SECTION AFTER STAGE 2 RESULTS ─────────────────
#
# TOP_CONFIGS: For each (dataset, horizon) condition, specify the best-performing
# GCN and TCN variant found in Stage 2.
#
# Format:
#   (dataset, horizon): (gcn_variant, tcn_variant, fusion_method)
#
# Example (update with your actual Stage 2 results):
#   ("metr-la",  3):  ("GWN",    "AMs",     "fgm"),
#   ("metr-la", 12):  ("GWN",    "EnStr",   "fgm"),
#   ("pems-bay", 3):  ("LightGAT", "AMs",   "fgm"),
#   ("pems-bay",12):  ("LightGAT", "EnStr", "fam"),
#
# NOTE: Keep fusion_method consistent with Stage 2 per-condition mapping.
# ─────────────────────────────────────────────────────────────────────────────
TOP_CONFIGS = {
    ("metr-la",   3): ("GWN",     "AMs",   "fgm"),
    ("metr-la",   6): ("GWN",     "AMs",   "fgm"),
    ("metr-la",  12): ("GWN",     "EnStr", "fgm"),
    ("pems-bay",  3): ("LightGAT","AMs",   "fgm"),
    ("pems-bay",  6): ("LightGAT","AMs",   "fgm"),
    ("pems-bay", 12): ("LightGAT","EnStr", "fam"),
}

# ─── VALIDATION SEEDS (seed=42 already done in Stage 2) ──────────────────────
EXTRA_SEEDS = [123, 456]

# ─── SAME HYPERPARAMETERS AS STAGES 1-3 ──────────────────────────────────────
_GCN_DIM       = 64
_TCN_DIM       = 64
_EPOCHS        = 100
_PATIENCE      = 15
_LEARNING_RATE = 1e-3      # (0.001) instead of 5e-4  — must stay as _LEARNING_RATE (used in _run_condition)
_WEIGHT_DECAY  = 1e-5
_DROPOUT       = 0.2
_BATCH_SIZE    = 128   # RTX 4090: 24GB VRAM
_NUM_WORKERS   = 8     # Linux server


def _run_condition(dataset: str, horizon: int, gcn: str, tcn: str, fusion: str, seeds: list) -> None:
    """Run one (dataset, horizon, gcn, tcn, fusion) config for the given seeds."""
    _t.DATASET         = f"[{dataset}]"
    _t.HISTORY         = 12
    _t.HORIZONS        = [horizon]
    _t.GCN_VARIANTS    = [gcn]
    _t.TCN_VARIANTS    = [tcn]
    _t.ARCHITECTURES   = ["parallel"]
    _t.FUSION_METHODS  = [fusion]
    _t.SEEDS           = seeds

    _t.GCN_DIM         = _GCN_DIM
    _t.TCN_DIM         = _TCN_DIM
    _t.EPOCHS          = _EPOCHS
    _t.PATIENCE        = _PATIENCE
    _t.LEARNING_RATE   = _LEARNING_RATE
    _t.WEIGHT_DECAY    = _WEIGHT_DECAY
    _t.DROPOUT         = _DROPOUT
    _t.BATCH_SIZE      = _BATCH_SIZE
    _t.NUM_WORKERS     = _NUM_WORKERS
    _t.LOG_VAL_METRICS      = True
    _t.SAVE_PREDS           = True   # save preds_test.npz for prediction-vs-GT figures
    _t.SLEEP_BETWEEN_RUNS_S = 1.0

    _t.main()


if __name__ == "__main__":
    print("=" * 80)
    n_new = len(TOP_CONFIGS) * len(EXTRA_SEEDS)
    print(f"SERVER STAGE 4: Multi-Seed Validation ({n_new} new experiments)")
    print(f"  Extra seeds: {EXTRA_SEEDS}  (seed=42 already exists from Stage 2)")
    print("=" * 80)

    for (dataset, horizon), (gcn, tcn, fusion) in TOP_CONFIGS.items():
        print(f"\n[Stage 4] {dataset} H={horizon} | {gcn}+{tcn} fusion={fusion} | seeds={EXTRA_SEEDS}")
        _run_condition(dataset, horizon, gcn, tcn, fusion, EXTRA_SEEDS)

    print("\n" + "=" * 80)
    print("Stage 4 complete. All multi-seed runs finished.")
    print("Run server_analyze.py to generate Table 12 (std/mean/CV statistics).")
    print("=" * 80)
