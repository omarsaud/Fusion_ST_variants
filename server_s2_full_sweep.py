"""
SERVER STAGE 2 — Full GCN × TCN Sweep  (THE MAIN EXPERIMENT)
=============================================================
Purpose : Evaluate all 6 GCN × 4 TCN combinations with the CORRECT fusion
          per condition (identified in Stage 1).
          Produces Tables 6, 7, 9 and Figures 20, 21.

  Excluded GCN: GL (redundant with GWN adaptive adjacency),
                SAGE (not designed for road-graph topology)
  Excluded TCN: GMs (subset of AMs gating mechanism)

CRITICAL DESIGN: Stage 2 runs TWO passes to use per-condition best fusion:
  Pass 2a (120 exp): FGM  ← best for METR-LA H=3/6/12 and PEMS-BAY H=3/6
  Pass 2b  (24 exp): FAM  ← best for PEMS-BAY H=12

  6 GCN × 4 TCN × 5 conditions (FGM) = 120
  6 GCN × 4 TCN × 1 condition  (FAM) =  24
  Total: 144 experiments

After Stage 1: Check H=6 fusion winners and update BEST_FUSION_MAP if needed.

save_preds=True is set for Pass 2b (PEMS-BAY H=12) to enable Figure 27.

Run from the 1912 parent folder:
  python Fusion_ST_variants/server_s2_full_sweep.py

Estimated time:
  RTX 3050 (6GB):  Pass 2a ~37-48h, Pass 2b ~9-12h  (Total ~46-60h)
  RTX 4090 (24GB): Pass 2a ~13-17h, Pass 2b ~3-4h   (Total ~16-21h, batch_size=128)
"""

from __future__ import annotations
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for p in [str(_HERE), str(_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import Fusion_ST_variants.train_ablation as _t

# ══════════════════════════════════════════════════════════════════════════════
# Per-condition fusion mapping (from Stage 1 screening):
# ══════════════════════════════════════════════════════════════════════════════
BEST_FUSION_MAP = {
    ("metr-la",   3): "fgm",
    ("metr-la",   6): "fgm",
    ("metr-la",  12): "fgm",
    ("pems-bay",  3): "fgm",
    ("pems-bay",  6): "fgm",
    ("pems-bay", 12): "fam",
}
# ══════════════════════════════════════════════════════════════════════════════

# GL excluded: same adaptive adjacency mechanism as GWN (Wu et al., IJCAI 2019)
# SAGE excluded: designed for general graphs, not road-network topology
# GMs excluded: sigmoid gating is a subset of AMs multi-head self-attention
ALL_GCN = ["GCN", "DiffusionGCN", "GWN", "GCN+AM", "GCN+Gating", "LightGAT"]  # 6 variants
ALL_TCN = ["Default", "STech", "EnStr", "AMs"]                                   # 4 variants

# ── Shared hyperparameters ─────────────────────────────────────────────────────
def _apply_hparams():
    _t.HISTORY       = 12
    _t.SEEDS         = [42]
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


def run_pass(label, dataset, horizons, fusion, save_preds_flag=False):
    """Run one sweep pass by overriding module-level variables."""
    _apply_hparams()
    _t.DATASET         = f"[{dataset}]"
    _t.HORIZONS        = horizons
    _t.GCN_VARIANTS    = ALL_GCN
    _t.TCN_VARIANTS    = ALL_TCN
    _t.ARCHITECTURES   = ["parallel"]
    _t.FUSION_METHODS  = [fusion]

    # Enable prediction saving via argparse default.
    # Patch _parse_args to inject save_preds when requested.
    import argparse as _ap
    _orig = _ap.ArgumentParser.parse_args
    if save_preds_flag:
        def _patched(self, *a, **kw):
            ns = _orig(self, *a, **kw)
            ns.save_preds = True
            return ns
        _ap.ArgumentParser.parse_args = _patched

    print("\n" + "=" * 80)
    print(f"STAGE 2 — {label}")
    print(f"  Dataset: {dataset} | Horizons: {horizons} | Fusion: {fusion}")
    print(f"  GCN×TCN: {len(ALL_GCN)}×{len(ALL_TCN)} = {len(ALL_GCN)*len(ALL_TCN)} combos")
    print(f"  Total exp: {len(ALL_GCN)*len(ALL_TCN)*len(horizons)}")
    print("=" * 80)
    _t.main()

    if save_preds_flag:
        _ap.ArgumentParser.parse_args = _orig


if __name__ == "__main__":
    # ── Pass 2a: FGM for 5 conditions (METR-LA H=3/6/12 + PEMS-BAY H=3/6) ────────
    # METR-LA: H=3, H=6, H=12
    run_pass("Pass 2a-1: METR-LA H=3/6/12 (FGM)", "metr-la",  [3, 6, 12], "fgm")

    # PEMS-BAY: H=3 and H=6
    run_pass("Pass 2a-2: PEMS-BAY H=3/6 (FGM)", "pems-bay", [3, 6], "fgm")

    # ── Pass 2b: FAM for PEMS-BAY H=12 ──────────────
    run_pass("Pass 2b: PEMS-BAY H=12 (FAM) — save_preds ON",
             "pems-bay", [12], "fam", save_preds_flag=True)

    print("\n" + "=" * 80)
    print("STAGE 2 COMPLETE — 144 experiments finished.")
    print("Results: Fusion_ST_variants_runs/results__metr-la.csv")
    print("         Fusion_ST_variants_runs/results__pems-bay.csv")
    print("=" * 80)
