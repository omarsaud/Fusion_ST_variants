"""
Stage 1 fusion selection: validation-loss ranking vs test-MAE ranking (no training).

Stage 1 (``server_s1_fusion_screen.py``) runs the four parallel fusions on four
encoder pairs per condition (GCN and GWN = Spectral GCN, each with the Default
and AMs TCN; seed 42). The original screen ranked fusions by mean test MAE.
This script re-ranks the same runs by mean validation loss (``best_val_loss``,
the L1 loss on Z-scored speed used for checkpoint selection) and reports, per
(dataset, horizon), the winner under each ranking. Runs with a non-finite
value are dropped (the GWN + FAM runs diverge). This is the source of
Supplementary Table S2 of the manuscript.

Usage (from the parent folder of Fusion_ST_variants_runs/):
    python Fusion_ST_variants/analysis/stage1_selection_check.py \
        --runs Fusion_ST_variants_runs
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="Fusion_ST_variants_runs",
                    help="folder with results__metr-la.csv and results__pems-bay.csv")
    args = ap.parse_args()

    agg: dict[tuple[str, int], dict[str, list[tuple[float, float]]]] = {}
    for ds in ("metr-la", "pems-bay"):
        with open(Path(args.runs) / f"results__{ds}.csv", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if (r["arch"] != "parallel" or r["seed"] != "42"
                        or r["gcn"] not in ("GCN", "GWN") or r["tcn"] not in ("Default", "AMs")):
                    continue
                try:
                    val, mae = float(r["best_val_loss"]), float(r["MAE"])
                except ValueError:
                    continue
                if not (math.isfinite(val) and math.isfinite(mae)) or mae > 50:
                    continue
                agg.setdefault((r["dataset"], int(r["horizon"])), {}).setdefault(r["fusion"], []).append((val, mae))

    same = 0
    for cond in sorted(agg):
        stats = {fu: (len(v), sum(x[0] for x in v) / len(v), sum(x[1] for x in v) / len(v))
                 for fu, v in agg[cond].items()}
        by_val = min(stats, key=lambda k: stats[k][1])
        by_test = min(stats, key=lambda k: stats[k][2])
        same += by_val == by_test
        print(f"\n{cond[0]}  H={cond[1]}")
        for fu in sorted(stats, key=lambda k: stats[k][1]):
            n, val, mae = stats[fu]
            print(f"  {fu:7s} n={n}  mean val loss={val:.4f}  mean test MAE={mae:.3f}")
        print(f"  winner by validation: {by_val} | winner by test: {by_test}")
    print(f"\nSame winner in {same} of {len(agg)} conditions.")


if __name__ == "__main__":
    main()
