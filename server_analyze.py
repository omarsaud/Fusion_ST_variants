"""
SERVER STAGE 5 — Post-Training Analysis & Export
=================================================
Purpose : Read all experiment CSV results and produce:
          ┌─────────────────────────────────────────────────────────────────┐
          │  TABLES (CSV + LaTeX)                                           │
          │    Table  6  — GCN variant comparison     (from Stage 2)        │
          │    Table  7  — TCN variant comparison     (from Stage 2)        │
          │    Table  8  — Architecture comparison    (from Stage 3)        │
          │    Table  9  — Fusion method comparison   (from Stage 2)        │
          │    Table 10  — Training cost              (from Stages 1-3)     │
          │    Table 11  — Model complexity           (parameter counts)    │
          │    Table 12  — Statistical robustness     (from Stage 4)        │
          ├─────────────────────────────────────────────────────────────────┤
          │  FIGURES (PNG + PDF)                                            │
          │    Figure 20 — GCN variant MAE comparison bars                  │
          │    Figure 21 — TCN variant MAE comparison bars                  │
          │    Figure 22 — Fusion method MAE comparison bars                │
          │    Figure 23 — FGM vs FAM deep-dive bars                        │
          │    Figure 24 — Parallel vs Stacked comparison                   │
          │    Figure 25 — Per-step MAE degradation (H=12 rollout)          │
          │    Figure 26 — Accuracy vs Training Cost scatter                │
          │    Figure 27 — Prediction vs Ground Truth  (requires save_preds)│
          └─────────────────────────────────────────────────────────────────┘

Run from the 1912 parent folder AFTER all training stages are complete:
  python Fusion_ST_variants/server_analyze.py

Outputs go to:  Fusion_ST_variants_runs/analysis/
  tables/   → Table_06.csv  Table_06.tex  ... Table_12.tex
  figures/  → fig20.png  fig20.pdf  ... fig27.png
"""

from __future__ import annotations

import sys
import math
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for p in [str(_HERE), str(_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

warnings.filterwarnings("ignore")

# ─── PATHS ────────────────────────────────────────────────────────────────────
RUNS_ROOT   = _ROOT / "Fusion_ST_variants_runs"
OUT_ROOT    = RUNS_ROOT / "analysis"
OUT_TABLES  = OUT_ROOT / "tables"
OUT_FIGURES = OUT_ROOT / "figures"

for d in [OUT_TABLES, OUT_FIGURES]:
    d.mkdir(parents=True, exist_ok=True)

# ─── PLOT STYLE ───────────────────────────────────────────────────────────────
DATASETS  = ["metr-la", "pems-bay"]
HORIZONS  = [3, 12]
COLOR_H3  = "#1565c0"   # deep blue  — H=3
COLOR_H12 = "#c62828"   # deep red   — H=12
HATCH_H3  = ""
HATCH_H12 = "///"
FIGSIZE_2P = (10.5, 3.8)  # 2-panel figure size

plt.rcParams.update({
    "font.family":     "DejaVu Sans",
    "font.size":       9,
    "axes.titlesize":  10,
    "axes.labelsize":  9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi":      150,
})


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════

def load_results() -> pd.DataFrame:
    """Load and merge all results CSVs from Fusion_ST_variants_runs/."""
    csv_files = list(RUNS_ROOT.glob("results__*.csv"))
    if not csv_files:
        print(f"[ERROR] No results CSV files found in {RUNS_ROOT}")
        print("        Make sure all training stages have completed before running analysis.")
        sys.exit(1)

    dfs = []
    for f in sorted(csv_files):
        df = pd.read_csv(f)
        dfs.append(df)
        print(f"  Loaded {f.name}: {len(df)} rows")

    df = pd.concat(dfs, ignore_index=True)

    # Normalise column names (strip whitespace, lowercase)
    df.columns = [c.strip() for c in df.columns]

    # Drop duplicates — keep last run (allows re-run override)
    key_cols = [c for c in ["dataset", "horizon", "gcn_variant", "tcn_variant",
                             "architecture", "fusion_method", "seed"] if c in df.columns]
    df = df.drop_duplicates(subset=key_cols, keep="last").reset_index(drop=True)

    print(f"\n  Total unique runs: {len(df)}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. LaTeX HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _bold(val: str) -> str:
    return f"\\textbf{{{val}}}"


def _fmt(v, decimals: int = 3) -> str:
    if pd.isna(v):
        return "—"
    return f"{float(v):.{decimals}f}"


def _save_latex(lines: List[str], name: str) -> None:
    path = OUT_TABLES / f"{name}.tex"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  [saved] {path.name}")


def _save_csv(df: pd.DataFrame, name: str) -> None:
    path = OUT_TABLES / f"{name}.csv"
    df.to_csv(path, index=False)
    print(f"  [saved] {path.name}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. TABLE 6 — GCN Variant Comparison
# ══════════════════════════════════════════════════════════════════════════════

GCN_ORDER = ["GCN", "DiffusionGCN", "GL", "SAGE", "GWN", "GCN+AM", "LightGAT", "GCN+Gating"]

def table_gcn(df: pd.DataFrame) -> pd.DataFrame:
    """
    Table 6: For each GCN variant, show MAE / RMSE / MAPE across 4 conditions
    (METR-LA H3, METR-LA H12, PEMS-BAY H3, PEMS-BAY H12).
    Uses parallel architecture, best fusion per condition (Stage 2 results).
    """
    print("\n[Table 6] GCN Variant Comparison")
    sub = df[
        (df["architecture"] == "parallel") &
        (df["seed"] == 42)
    ].copy()

    records = []
    for gcn in GCN_ORDER:
        g = sub[sub["gcn_variant"] == gcn]
        if g.empty:
            print(f"  [warn] No data for GCN={gcn}")
        row = {"GCN Variant": gcn}
        for ds in DATASETS:
            for h in HORIZONS:
                cond = g[(g["dataset"] == ds) & (g["horizon"] == h)]
                # If multiple fusions, pick the best MAE
                if not cond.empty:
                    best = cond.loc[cond["MAE"].idxmin()]
                    row[f"{ds}_H{h}_MAE"]  = best["MAE"]
                    row[f"{ds}_H{h}_RMSE"] = best["RMSE"]
                    row[f"{ds}_H{h}_MAPE"] = best["MAPE"]
                    row[f"{ds}_H{h}_fusion"] = best.get("fusion_method", "")
                else:
                    row[f"{ds}_H{h}_MAE"]  = float("nan")
                    row[f"{ds}_H{h}_RMSE"] = float("nan")
                    row[f"{ds}_H{h}_MAPE"] = float("nan")
                    row[f"{ds}_H{h}_fusion"] = ""
        records.append(row)

    t6 = pd.DataFrame(records)
    _save_csv(t6, "Table_06_gcn_variants")

    # LaTeX
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{GCN variant comparison across datasets and prediction horizons "
        r"(parallel architecture, best fusion per condition, seed=42).}",
        r"\label{tab:gcn_variants}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l|ccc|ccc|ccc|ccc}",
        r"\toprule",
        r" & \multicolumn{3}{c|}{METR-LA $H=3$} & \multicolumn{3}{c|}{METR-LA $H=12$}"
        r" & \multicolumn{3}{c|}{PEMS-BAY $H=3$} & \multicolumn{3}{c}{PEMS-BAY $H=12$} \\",
        r"GCN Variant & MAE & RMSE & MAPE & MAE & RMSE & MAPE & MAE & RMSE & MAPE & MAE & RMSE & MAPE \\",
        r"\midrule",
    ]
    for _, row in t6.iterrows():
        cells = [str(row["GCN Variant"])]
        for ds in DATASETS:
            for h in HORIZONS:
                cells += [
                    _fmt(row.get(f"{ds}_H{h}_MAE")),
                    _fmt(row.get(f"{ds}_H{h}_RMSE")),
                    _fmt(row.get(f"{ds}_H{h}_MAPE"), decimals=2),
                ]
        lines.append(" & ".join(cells) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    _save_latex(lines, "Table_06_gcn_variants")
    return t6


# ══════════════════════════════════════════════════════════════════════════════
# 4. TABLE 7 — TCN Variant Comparison
# ══════════════════════════════════════════════════════════════════════════════

TCN_ORDER = ["Default", "EnStr", "AMs", "GMs"]

def table_tcn(df: pd.DataFrame) -> pd.DataFrame:
    """Table 7: TCN variant comparison (parallel arch, best fusion, seed=42)."""
    print("\n[Table 7] TCN Variant Comparison")
    sub = df[
        (df["architecture"] == "parallel") &
        (df["seed"] == 42)
    ].copy()

    records = []
    for tcn in TCN_ORDER:
        g = sub[sub["tcn_variant"] == tcn]
        row = {"TCN Variant": tcn}
        for ds in DATASETS:
            for h in HORIZONS:
                cond = g[(g["dataset"] == ds) & (g["horizon"] == h)]
                if not cond.empty:
                    best = cond.loc[cond["MAE"].idxmin()]
                    row[f"{ds}_H{h}_MAE"]  = best["MAE"]
                    row[f"{ds}_H{h}_RMSE"] = best["RMSE"]
                    row[f"{ds}_H{h}_MAPE"] = best["MAPE"]
                else:
                    row[f"{ds}_H{h}_MAE"]  = float("nan")
                    row[f"{ds}_H{h}_RMSE"] = float("nan")
                    row[f"{ds}_H{h}_MAPE"] = float("nan")
        records.append(row)

    t7 = pd.DataFrame(records)
    _save_csv(t7, "Table_07_tcn_variants")

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{TCN variant comparison across datasets and prediction horizons "
        r"(parallel architecture, best fusion per condition, seed=42).}",
        r"\label{tab:tcn_variants}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l|ccc|ccc|ccc|ccc}",
        r"\toprule",
        r" & \multicolumn{3}{c|}{METR-LA $H=3$} & \multicolumn{3}{c|}{METR-LA $H=12$}"
        r" & \multicolumn{3}{c|}{PEMS-BAY $H=3$} & \multicolumn{3}{c}{PEMS-BAY $H=12$} \\",
        r"TCN Variant & MAE & RMSE & MAPE & MAE & RMSE & MAPE & MAE & RMSE & MAPE & MAE & RMSE & MAPE \\",
        r"\midrule",
    ]
    for _, row in t7.iterrows():
        cells = [str(row["TCN Variant"])]
        for ds in DATASETS:
            for h in HORIZONS:
                cells += [
                    _fmt(row.get(f"{ds}_H{h}_MAE")),
                    _fmt(row.get(f"{ds}_H{h}_RMSE")),
                    _fmt(row.get(f"{ds}_H{h}_MAPE"), decimals=2),
                ]
        lines.append(" & ".join(cells) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    _save_latex(lines, "Table_07_tcn_variants")
    return t7


# ══════════════════════════════════════════════════════════════════════════════
# 5. TABLE 8 — Architecture Comparison (Parallel vs Stacked)
# ══════════════════════════════════════════════════════════════════════════════

def table_architecture(df: pd.DataFrame) -> pd.DataFrame:
    """Table 8: Architecture comparison. Parallel (best fusion) vs Stack (direct)."""
    print("\n[Table 8] Architecture Comparison (Parallel vs Stack)")

    records = []
    for gcn in GCN_ORDER:
        row = {"GCN Variant": gcn}
        for ds in DATASETS:
            for h in HORIZONS:
                # Parallel — best fusion
                p = df[
                    (df["gcn_variant"] == gcn) &
                    (df["architecture"] == "parallel") &
                    (df["dataset"] == ds) &
                    (df["horizon"] == h) &
                    (df["seed"] == 42)
                ]
                if not p.empty:
                    best_p = p.loc[p["MAE"].idxmin()]
                    row[f"{ds}_H{h}_parallel_MAE"] = best_p["MAE"]
                else:
                    row[f"{ds}_H{h}_parallel_MAE"] = float("nan")

                # Stack — direct fusion
                s = df[
                    (df["gcn_variant"] == gcn) &
                    (df["architecture"] == "stack") &
                    (df["dataset"] == ds) &
                    (df["horizon"] == h) &
                    (df["seed"] == 42)
                ]
                if not s.empty:
                    row[f"{ds}_H{h}_stack_MAE"] = float(s.iloc[0]["MAE"])
                else:
                    row[f"{ds}_H{h}_stack_MAE"] = float("nan")

                # Advantage %
                pv = row.get(f"{ds}_H{h}_parallel_MAE", float("nan"))
                sv = row.get(f"{ds}_H{h}_stack_MAE", float("nan"))
                if not (math.isnan(pv) or math.isnan(sv) or sv == 0):
                    row[f"{ds}_H{h}_adv_pct"] = 100.0 * (sv - pv) / sv
                else:
                    row[f"{ds}_H{h}_adv_pct"] = float("nan")

        records.append(row)

    t8 = pd.DataFrame(records)
    _save_csv(t8, "Table_08_architecture")

    # LaTeX — compact: Parallel MAE | Stack MAE | Adv%
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Architecture comparison: parallel fusion vs.\ stacked (GCN$\to$TCN) "
        r"baseline. Adv.\ = percentage MAE reduction of parallel over stacked.}",
        r"\label{tab:architecture}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l|ccc|ccc|ccc|ccc}",
        r"\toprule",
        r" & \multicolumn{3}{c|}{METR-LA $H=3$} & \multicolumn{3}{c|}{METR-LA $H=12$}"
        r" & \multicolumn{3}{c|}{PEMS-BAY $H=3$} & \multicolumn{3}{c}{PEMS-BAY $H=12$} \\",
        r"GCN & Par. & Stk. & Adv.\% & Par. & Stk. & Adv.\% & Par. & Stk. & Adv.\%"
        r" & Par. & Stk. & Adv.\% \\",
        r"\midrule",
    ]
    for _, row in t8.iterrows():
        cells = [str(row["GCN Variant"])]
        for ds in DATASETS:
            for h in HORIZONS:
                cells += [
                    _fmt(row.get(f"{ds}_H{h}_parallel_MAE")),
                    _fmt(row.get(f"{ds}_H{h}_stack_MAE")),
                    _fmt(row.get(f"{ds}_H{h}_adv_pct"), decimals=1),
                ]
        lines.append(" & ".join(cells) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    _save_latex(lines, "Table_08_architecture")
    return t8


# ══════════════════════════════════════════════════════════════════════════════
# 6. TABLE 9 — Fusion Method Comparison
# ══════════════════════════════════════════════════════════════════════════════

FUSION_ORDER = ["direct", "fgm", "fam", "pmf"]

def table_fusion(df: pd.DataFrame) -> pd.DataFrame:
    """Table 9: Fusion method comparison (parallel arch, GCN=GWN best GCN, seed=42)."""
    print("\n[Table 9] Fusion Method Comparison")

    # Use GWN as the representative GCN (typically best performer)
    # Fall back to any GCN if GWN has no data
    sub = df[
        (df["architecture"] == "parallel") &
        (df["seed"] == 42)
    ].copy()

    records = []
    for fusion in FUSION_ORDER:
        g = sub[sub["fusion_method"] == fusion]
        row = {"Fusion Method": fusion.upper()}
        for ds in DATASETS:
            for h in HORIZONS:
                cond = g[(g["dataset"] == ds) & (g["horizon"] == h)]
                if not cond.empty:
                    # Average over all GCN/TCN combos (shows fusion effect)
                    row[f"{ds}_H{h}_MAE"]  = cond["MAE"].mean()
                    row[f"{ds}_H{h}_RMSE"] = cond["RMSE"].mean()
                    row[f"{ds}_H{h}_MAPE"] = cond["MAPE"].mean()
                else:
                    row[f"{ds}_H{h}_MAE"]  = float("nan")
                    row[f"{ds}_H{h}_RMSE"] = float("nan")
                    row[f"{ds}_H{h}_MAPE"] = float("nan")
        records.append(row)

    t9 = pd.DataFrame(records)
    _save_csv(t9, "Table_09_fusion_methods")

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Fusion method comparison (parallel architecture, values averaged "
        r"over all GCN$\times$TCN combinations, seed=42).}",
        r"\label{tab:fusion_methods}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l|ccc|ccc|ccc|ccc}",
        r"\toprule",
        r" & \multicolumn{3}{c|}{METR-LA $H=3$} & \multicolumn{3}{c|}{METR-LA $H=12$}"
        r" & \multicolumn{3}{c|}{PEMS-BAY $H=3$} & \multicolumn{3}{c}{PEMS-BAY $H=12$} \\",
        r"Fusion & MAE & RMSE & MAPE & MAE & RMSE & MAPE & MAE & RMSE & MAPE & MAE & RMSE & MAPE \\",
        r"\midrule",
    ]
    for _, row in t9.iterrows():
        cells = [str(row["Fusion Method"])]
        for ds in DATASETS:
            for h in HORIZONS:
                cells += [
                    _fmt(row.get(f"{ds}_H{h}_MAE")),
                    _fmt(row.get(f"{ds}_H{h}_RMSE")),
                    _fmt(row.get(f"{ds}_H{h}_MAPE"), decimals=2),
                ]
        lines.append(" & ".join(cells) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    _save_latex(lines, "Table_09_fusion_methods")
    return t9


# ══════════════════════════════════════════════════════════════════════════════
# 7. TABLE 10 — Training Cost
# ══════════════════════════════════════════════════════════════════════════════

def table_training_cost(df: pd.DataFrame) -> pd.DataFrame:
    """Table 10: Training time per GCN variant (METR-LA H=12, parallel, seed=42)."""
    print("\n[Table 10] Training Cost")

    sub = df[
        (df["dataset"] == "metr-la") &
        (df["horizon"] == 12) &
        (df["architecture"] == "parallel") &
        (df["seed"] == 42)
    ]

    records = []
    for gcn in GCN_ORDER:
        g = sub[sub["gcn_variant"] == gcn]
        if g.empty:
            continue
        best = g.loc[g["MAE"].idxmin()]
        records.append({
            "GCN Variant":         gcn,
            "Total Train (min)":   round(float(best.get("train_time_s", float("nan"))) / 60, 1),
            "Avg Epoch (s)":       round(float(best.get("avg_epoch_time_s", float("nan"))), 1),
            "Best Epoch":          int(best.get("best_epoch", -1)),
            "Epochs Run":          int(best.get("epochs_ran", -1)),
            "MAE (METR-LA H=12)":  round(float(best.get("MAE", float("nan"))), 3),
        })

    t10 = pd.DataFrame(records)
    _save_csv(t10, "Table_10_training_cost")

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Training cost by GCN variant (METR-LA, $H=12$, parallel architecture, "
        r"seed=42). Training conducted on NVIDIA RTX 3050 with GCN\_dim=64, TCN\_dim=128.}",
        r"\label{tab:training_cost}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"GCN Variant & Train (min) & Epoch (s) & Best Epoch & Epochs Run & MAE \\",
        r"\midrule",
    ]
    for _, row in t10.iterrows():
        cells = [
            str(row["GCN Variant"]),
            _fmt(row["Total Train (min)"], decimals=1),
            _fmt(row["Avg Epoch (s)"], decimals=1),
            str(int(row["Best Epoch"])) if row["Best Epoch"] >= 0 else "—",
            str(int(row["Epochs Run"])) if row["Epochs Run"] >= 0 else "—",
            _fmt(row["MAE (METR-LA H=12)"]),
        ]
        lines.append(" & ".join(cells) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _save_latex(lines, "Table_10_training_cost")
    return t10


# ══════════════════════════════════════════════════════════════════════════════
# 8. TABLE 11 — Model Complexity (Parameter Count)
# ══════════════════════════════════════════════════════════════════════════════

def _count_params_gcn(gcn_name: str, gcn_dim: int = 64) -> int:
    """Estimate parameter count for each GCN variant (input_dim=1 → gcn_dim)."""
    n = gcn_name.strip().lower()
    d = gcn_dim
    if n in {"gcn", "d-gcn", "default", "diffusiongcn", "dgcn", "diffusion",
             "gl", "graphadaptivelearning", "adaptive", "sage", "graphsage",
             "gwn", "graphwaveletnetwork", "wavelet"}:
        # Linear: 1→d + bias = d + d = 2d
        # SAGE: (1+1)→d + bias = 2d + d = 3d
        if n in {"sage", "graphsage"}:
            return 2 * d + d          # fc: [2,d] + bias [d]
        return 1 * d + d              # fc: [1,d] + bias [d]

    if n in {"gcn+am", "am", "attention"}:
        # query+key+value each: d×d, fc: d×d, BN: 2d → ~4d^2 + 2d
        return 4 * d * d + 2 * d

    if n in {"gcn+gating", "gating", "gate"}:
        # feature_gate + update_gate + output_fc ~ 3d^2
        return 3 * d * d + 2 * d

    if n in {"lightgat", "gat"}:
        # fc: d×d + attn_proj: d×1 ~ d^2 + d
        return d * d + d

    return d * d  # fallback


def _count_params_tcn(tcn_name: str, tcn_dim: int = 128, gcn_dim: int = 64) -> int:
    """Estimate parameter count for each TCN variant."""
    n = tcn_name.strip().lower()
    d = tcn_dim
    inp = gcn_dim  # For stack arch; for parallel arch input_dim=1

    if n in {"default", "d-tcn"}:
        # 2-layer causal conv: inp→d→d, kernel_size=3
        return (inp * d * 3 + d) + (d * d * 3 + d)

    if n in {"enstr", "enhanced", "enhanced_stride"}:
        # Multi-layer with increasing dilations; approx 4 layers
        return 4 * (d * d * 3 + d) + 4 * d  # 4×conv + 4×LayerNorm

    if n in {"ams", "attention_ms"}:
        # DefaultTCN + MultiheadAttention
        base = (1 * d * 3 + d) + (d * d * 3 + d)  # default tcn (input=1 for parallel)
        attn = 4 * d * d  # Q, K, V, O projections
        return base + attn

    if n in {"gms", "gated_ms"}:
        # DefaultTCN + sigmoid gate
        base = (1 * d * 3 + d) + (d * d * 3 + d)
        gate = d * d + d
        return base + gate

    return d * d  # fallback


def table_model_complexity(df: pd.DataFrame) -> pd.DataFrame:
    """Table 11: Model parameter counts for each GCN+TCN combination."""
    print("\n[Table 11] Model Complexity")

    records = []
    gcn_dim, tcn_dim = 64, 128

    for gcn in GCN_ORDER:
        for tcn in TCN_ORDER:
            gcn_p  = _count_params_gcn(gcn, gcn_dim)
            tcn_p  = _count_params_tcn(tcn, tcn_dim, gcn_dim)
            head_p = (gcn_dim + tcn_dim) * 12 + 12  # head: combined→horizon

            total = gcn_p + tcn_p + head_p
            records.append({
                "GCN Variant": gcn,
                "TCN Variant": tcn,
                "GCN Params":  gcn_p,
                "TCN Params":  tcn_p,
                "Head Params": head_p,
                "Total Params": total,
                "Total (K)":   round(total / 1000, 1),
            })

    t11 = pd.DataFrame(records)
    _save_csv(t11, "Table_11_model_complexity")

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Model complexity: estimated parameter counts for each GCN$\times$TCN "
        r"combination (parallel architecture, GCN\_dim=64, TCN\_dim=128, horizon=12).}",
        r"\label{tab:model_complexity}",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"GCN Variant & TCN Variant & GCN & TCN & Head & Total & Total (K) \\",
        r"\midrule",
    ]
    for _, row in t11.iterrows():
        cells = [
            str(row["GCN Variant"]),
            str(row["TCN Variant"]),
            f"{int(row['GCN Params']):,}",
            f"{int(row['TCN Params']):,}",
            f"{int(row['Head Params']):,}",
            f"{int(row['Total Params']):,}",
            f"{row['Total (K)']:.1f}",
        ]
        lines.append(" & ".join(cells) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _save_latex(lines, "Table_11_model_complexity")
    return t11


# ══════════════════════════════════════════════════════════════════════════════
# 9. TABLE 12 — Statistical Robustness (Multi-Seed)
# ══════════════════════════════════════════════════════════════════════════════

def table_statistical_robustness(df: pd.DataFrame) -> pd.DataFrame:
    """
    Table 12: For each top config per condition, report MAE mean ± std across seeds.
    Requires Stage 4 to have completed (seeds 42, 123, 456).
    """
    print("\n[Table 12] Statistical Robustness (Multi-Seed)")

    # Find configs that have multiple seeds
    key_cols = ["dataset", "horizon", "gcn_variant", "tcn_variant", "fusion_method"]
    multi = df[df["architecture"] == "parallel"].copy()

    # Group by config, count seeds
    grouped = multi.groupby(key_cols)["seed"].count().reset_index()
    multi_seed_configs = grouped[grouped["seed"] > 1]

    if multi_seed_configs.empty:
        print("  [warn] No multi-seed configurations found. Stage 4 may not be complete.")
        t12 = pd.DataFrame(columns=["Config", "Dataset", "H", "Seeds",
                                     "MAE mean", "MAE std", "CV%"])
        _save_csv(t12, "Table_12_robustness")
        return t12

    records = []
    for _, cfg in multi_seed_configs.iterrows():
        cond = multi.copy()
        for col in key_cols:
            cond = cond[cond[col] == cfg[col]]

        mae_vals = cond["MAE"].values
        mean_mae = float(np.mean(mae_vals))
        std_mae  = float(np.std(mae_vals, ddof=1)) if len(mae_vals) > 1 else 0.0
        cv_pct   = 100.0 * std_mae / mean_mae if mean_mae > 0 else 0.0

        records.append({
            "GCN": cfg["gcn_variant"],
            "TCN": cfg["tcn_variant"],
            "Fusion": cfg["fusion_method"],
            "Dataset": cfg["dataset"],
            "H": int(cfg["horizon"]),
            "Seeds": int(cfg["seed"]),
            "MAE mean": round(mean_mae, 4),
            "MAE std":  round(std_mae, 4),
            "CV%":      round(cv_pct, 2),
        })

    t12 = pd.DataFrame(records).sort_values(["Dataset", "H", "MAE mean"])
    _save_csv(t12, "Table_12_robustness")

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Statistical robustness of top-performing configurations across "
        r"multiple random seeds (parallel architecture). CV = coefficient of variation.}",
        r"\label{tab:robustness}",
        r"\begin{tabular}{llllccrrr}",
        r"\toprule",
        r"GCN & TCN & Fusion & Dataset & H & Seeds & MAE mean & MAE std & CV\% \\",
        r"\midrule",
    ]
    for _, row in t12.iterrows():
        cells = [
            str(row["GCN"]),
            str(row["TCN"]),
            str(row["Fusion"]),
            str(row["Dataset"]),
            str(int(row["H"])),
            str(int(row["Seeds"])),
            _fmt(row["MAE mean"], decimals=4),
            _fmt(row["MAE std"], decimals=4),
            _fmt(row["CV%"], decimals=2),
        ]
        lines.append(" & ".join(cells) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _save_latex(lines, "Table_12_robustness")
    return t12


# ══════════════════════════════════════════════════════════════════════════════
# 10. FIGURE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _savefig(fig: plt.Figure, name: str) -> None:
    for ext in ["png", "pdf"]:
        p = OUT_FIGURES / f"{name}.{ext}"
        fig.savefig(str(p), bbox_inches="tight", dpi=150)
    print(f"  [saved] {name}.png / .pdf")
    plt.close(fig)


def _mae_bars(ax: plt.Axes, labels: List[str],
              vals_h3: List[float], vals_h12: List[float],
              baseline_h3: Optional[float] = None,
              baseline_h12: Optional[float] = None,
              title: str = "") -> None:
    """Draw grouped bars (H3=blue solid, H12=red hatched) with optional baselines."""
    x = np.arange(len(labels))
    w = 0.35

    bars3  = ax.bar(x - w/2, vals_h3,  w, color=COLOR_H3,  hatch=HATCH_H3,
                    edgecolor="white", label="H=3")
    bars12 = ax.bar(x + w/2, vals_h12, w, color=COLOR_H12, hatch=HATCH_H12,
                    edgecolor="white", alpha=0.85, label="H=12")

    if baseline_h3 is not None and not math.isnan(baseline_h3):
        ax.axhline(baseline_h3, color=COLOR_H3,  linestyle="--", linewidth=1.0,
                   alpha=0.7, label=f"Baseline H=3 ({baseline_h3:.3f})")
    if baseline_h12 is not None and not math.isnan(baseline_h12):
        ax.axhline(baseline_h12, color=COLOR_H12, linestyle="--", linewidth=1.0,
                   alpha=0.7, label=f"Baseline H=12 ({baseline_h12:.3f})")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("MAE")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, ncol=2)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.grid(axis="y", linestyle=":", alpha=0.5)


def _get_mae_series(df: pd.DataFrame, group_col: str, order: List[str],
                    ds: str, filt: dict) -> Tuple[List[float], List[float]]:
    """Extract MAE for H=3 and H=12 for each value in `order`."""
    sub = df.copy()
    for k, v in filt.items():
        sub = sub[sub[k] == v]
    sub = sub[sub["dataset"] == ds]

    vals3, vals12 = [], []
    for label in order:
        g = sub[sub[group_col] == label]
        for h, lst in [(3, vals3), (12, vals12)]:
            row = g[g["horizon"] == h]
            if not row.empty:
                lst.append(float(row.loc[row["MAE"].idxmin(), "MAE"]))
            else:
                lst.append(float("nan"))
    return vals3, vals12


# ══════════════════════════════════════════════════════════════════════════════
# 11. FIGURES 20–24
# ══════════════════════════════════════════════════════════════════════════════

def fig_gcn(df: pd.DataFrame) -> None:
    """Figure 20: GCN variant MAE comparison (parallel, seed=42)."""
    print("\n[Figure 20] GCN Variant Comparison")
    sub = df[(df["architecture"] == "parallel") & (df["seed"] == 42)]

    # Baseline = GCN (D-GCN)
    bl = {}
    for ds in DATASETS:
        for h in HORIZONS:
            r = sub[(sub["gcn_variant"] == "GCN") & (sub["dataset"] == ds) & (sub["horizon"] == h)]
            bl[(ds, h)] = float(r["MAE"].min()) if not r.empty else float("nan")

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_2P)
    for ax, ds in zip(axes, DATASETS):
        v3, v12 = _get_mae_series(sub, "gcn_variant", GCN_ORDER, ds,
                                   {"architecture": "parallel", "seed": 42})
        _mae_bars(ax, GCN_ORDER, v3, v12,
                  baseline_h3=bl.get((ds, 3)), baseline_h12=bl.get((ds, 12)),
                  title=f"GCN Variants — {ds.upper()}")
    plt.tight_layout()
    _savefig(fig, "fig20_gcn_variants")


def fig_tcn(df: pd.DataFrame) -> None:
    """Figure 21: TCN variant MAE comparison (parallel, seed=42)."""
    print("\n[Figure 21] TCN Variant Comparison")
    sub = df[(df["architecture"] == "parallel") & (df["seed"] == 42)]

    bl = {}
    for ds in DATASETS:
        for h in HORIZONS:
            r = sub[(sub["tcn_variant"] == "Default") & (sub["dataset"] == ds) & (sub["horizon"] == h)]
            bl[(ds, h)] = float(r["MAE"].min()) if not r.empty else float("nan")

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_2P)
    for ax, ds in zip(axes, DATASETS):
        v3, v12 = _get_mae_series(sub, "tcn_variant", TCN_ORDER, ds,
                                   {"architecture": "parallel", "seed": 42})
        _mae_bars(ax, TCN_ORDER, v3, v12,
                  baseline_h3=bl.get((ds, 3)), baseline_h12=bl.get((ds, 12)),
                  title=f"TCN Variants — {ds.upper()}")
    plt.tight_layout()
    _savefig(fig, "fig21_tcn_variants")


def fig_fusion(df: pd.DataFrame) -> None:
    """Figure 22: Fusion method MAE comparison (parallel, seed=42)."""
    print("\n[Figure 22] Fusion Method Comparison")
    sub = df[(df["architecture"] == "parallel") & (df["seed"] == 42)]

    bl = {}
    for ds in DATASETS:
        for h in HORIZONS:
            r = sub[(sub["fusion_method"] == "direct") & (sub["dataset"] == ds) & (sub["horizon"] == h)]
            bl[(ds, h)] = float(r["MAE"].mean()) if not r.empty else float("nan")

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_2P)
    for ax, ds in zip(axes, DATASETS):
        v3, v12 = [], []
        for fusion in FUSION_ORDER:
            for h, lst in [(3, v3), (12, v12)]:
                r = sub[(sub["fusion_method"] == fusion) & (sub["dataset"] == ds) & (sub["horizon"] == h)]
                lst.append(float(r["MAE"].mean()) if not r.empty else float("nan"))
        labels = [f.upper() for f in FUSION_ORDER]
        _mae_bars(ax, labels, v3, v12,
                  baseline_h3=bl.get((ds, 3)), baseline_h12=bl.get((ds, 12)),
                  title=f"Fusion Methods — {ds.upper()}")
    plt.tight_layout()
    _savefig(fig, "fig22_fusion_methods")


def fig_fgm_fam(df: pd.DataFrame) -> None:
    """Figure 23: FGM vs FAM deep-dive per GCN (parallel, seed=42)."""
    print("\n[Figure 23] FGM vs FAM Deep-Dive")
    sub = df[
        (df["architecture"] == "parallel") &
        (df["seed"] == 42) &
        (df["fusion_method"].isin(["fgm", "fam"]))
    ]

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_2P)
    for ax, ds in zip(axes, DATASETS):
        x = np.arange(len(GCN_ORDER))
        w = 0.2
        offsets = [-1.5*w, -0.5*w, 0.5*w, 1.5*w]
        colors_fusions = {
            ("fgm", 3):  (COLOR_H3,  HATCH_H3,  "FGM H=3"),
            ("fgm", 12): (COLOR_H3,  HATCH_H12, "FGM H=12"),
            ("fam", 3):  (COLOR_H12, HATCH_H3,  "FAM H=3"),
            ("fam", 12): (COLOR_H12, HATCH_H12, "FAM H=12"),
        }
        for (fusion, h), (col, hatch, lbl) in zip(
                colors_fusions.keys(),
                colors_fusions.values()):
            vals = []
            for gcn in GCN_ORDER:
                r = sub[
                    (sub["gcn_variant"] == gcn) & (sub["dataset"] == ds) &
                    (sub["horizon"] == h) & (sub["fusion_method"] == fusion)
                ]
                vals.append(float(r["MAE"].min()) if not r.empty else float("nan"))
            off = offsets[list(colors_fusions.keys()).index((fusion, h))]
            ax.bar(x + off, vals, w, color=col, hatch=hatch,
                   edgecolor="white", alpha=0.85, label=lbl)

        ax.set_xticks(x)
        ax.set_xticklabels(GCN_ORDER, rotation=30, ha="right", fontsize=7)
        ax.set_ylabel("MAE")
        ax.set_title(f"FGM vs FAM — {ds.upper()}", fontsize=10)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(axis="y", linestyle=":", alpha=0.5)

    plt.tight_layout()
    _savefig(fig, "fig23_fgm_fam")


def fig_parallel_stacked(df: pd.DataFrame) -> None:
    """Figure 24: Parallel vs Stacked (direct) per GCN variant."""
    print("\n[Figure 24] Parallel vs Stacked")
    sub = df[df["seed"] == 42]

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_2P)
    for ax, ds in zip(axes, DATASETS):
        x = np.arange(len(GCN_ORDER))
        w = 0.2
        for i, (arch, fusion, h, col, hatch, lbl) in enumerate([
            ("parallel", None,     3,  COLOR_H3,  HATCH_H3,  "Par H=3"),
            ("parallel", None,    12,  COLOR_H3,  HATCH_H12, "Par H=12"),
            ("stack",   "direct",  3,  COLOR_H12, HATCH_H3,  "Stk H=3"),
            ("stack",   "direct", 12,  COLOR_H12, HATCH_H12, "Stk H=12"),
        ]):
            vals = []
            for gcn in GCN_ORDER:
                r = sub[(sub["gcn_variant"] == gcn) & (sub["dataset"] == ds) &
                        (sub["horizon"] == h) & (sub["architecture"] == arch)]
                if fusion:
                    r = r[r["fusion_method"] == fusion]
                vals.append(float(r["MAE"].min()) if not r.empty else float("nan"))
            off = (i - 1.5) * w
            ax.bar(x + off, vals, w, color=col, hatch=hatch,
                   edgecolor="white", alpha=0.85, label=lbl)

        ax.set_xticks(x)
        ax.set_xticklabels(GCN_ORDER, rotation=30, ha="right", fontsize=7)
        ax.set_ylabel("MAE")
        ax.set_title(f"Parallel vs Stacked — {ds.upper()}", fontsize=10)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(axis="y", linestyle=":", alpha=0.5)

    plt.tight_layout()
    _savefig(fig, "fig24_parallel_stacked")


# ══════════════════════════════════════════════════════════════════════════════
# 12. FIGURE 25 — Per-Step MAE Degradation (H=12 rollout)
# ══════════════════════════════════════════════════════════════════════════════

def fig_perstep(df: pd.DataFrame) -> None:
    """Figure 25: Per-step MAE at steps 3, 6, 12 for each GCN variant (H=12)."""
    print("\n[Figure 25] Per-Step MAE Degradation")

    step_cols = ["MAE@3", "MAE@6", "MAE@12"]
    missing = [c for c in step_cols if c not in df.columns]
    if missing:
        print(f"  [warn] Per-step columns not found: {missing}. Skipping Figure 25.")
        return

    sub = df[
        (df["architecture"] == "parallel") &
        (df["horizon"] == 12) &
        (df["seed"] == 42)
    ]

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_2P)
    steps = [3, 6, 12]
    cmap  = plt.get_cmap("tab10")

    for ax, ds in zip(axes, DATASETS):
        s = sub[sub["dataset"] == ds]
        for i, gcn in enumerate(GCN_ORDER):
            g = s[s["gcn_variant"] == gcn]
            if g.empty:
                continue
            best = g.loc[g["MAE"].idxmin()]
            vals = [float(best.get(f"MAE@{st}", float("nan"))) for st in steps]
            ax.plot(steps, vals, marker="o", linewidth=1.5,
                    color=cmap(i), label=gcn, alpha=0.85)

        ax.set_xlabel("Prediction Step")
        ax.set_ylabel("MAE")
        ax.set_xticks(steps)
        ax.set_title(f"Per-Step MAE Degradation (H=12) — {ds.upper()}", fontsize=10)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(linestyle=":", alpha=0.5)

    plt.tight_layout()
    _savefig(fig, "fig25_perstep_mae")


# ══════════════════════════════════════════════════════════════════════════════
# 13. FIGURE 26 — Accuracy vs Training Cost
# ══════════════════════════════════════════════════════════════════════════════

def fig_accuracy_vs_cost(df: pd.DataFrame) -> None:
    """Figure 26: Scatter of MAE vs Training Time (METR-LA, H=12, parallel)."""
    print("\n[Figure 26] Accuracy vs Training Cost")

    if "train_time_s" not in df.columns:
        print("  [warn] 'train_time_s' not in data. Skipping Figure 26.")
        return

    sub = df[
        (df["dataset"] == "metr-la") &
        (df["horizon"] == 12) &
        (df["architecture"] == "parallel") &
        (df["seed"] == 42)
    ]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    cmap = plt.get_cmap("tab10")

    for i, gcn in enumerate(GCN_ORDER):
        g = sub[sub["gcn_variant"] == gcn]
        if g.empty:
            continue
        for _, row in g.iterrows():
            mae = float(row.get("MAE", float("nan")))
            cost = float(row.get("train_time_s", float("nan"))) / 60.0
            if not (math.isnan(mae) or math.isnan(cost)):
                ax.scatter(cost, mae, color=cmap(i), s=60, alpha=0.8, label=gcn
                           if not any(l.get_label() == gcn for l in ax.collections) else "")

    ax.set_xlabel("Training Time (minutes)")
    ax.set_ylabel("MAE")
    ax.set_title("Accuracy vs Training Cost (METR-LA, H=12, Parallel)")

    # Deduplicate legend
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    unique_h, unique_l = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l)
            unique_h.append(h)
            unique_l.append(l)
    ax.legend(unique_h, unique_l, fontsize=8, ncol=2, loc="upper right")
    ax.grid(linestyle=":", alpha=0.5)

    plt.tight_layout()
    _savefig(fig, "fig26_accuracy_vs_cost")


# ══════════════════════════════════════════════════════════════════════════════
# 14. FIGURE 27 — Prediction vs Ground Truth
# ══════════════════════════════════════════════════════════════════════════════

def fig_pred_vs_gt(df: pd.DataFrame) -> None:
    """
    Figure 27: Prediction vs Ground Truth time-series plot.
    Requires save_preds=True from Stage 2 Pass 2b.
    Looks for predictions.npz files in run directories.
    """
    print("\n[Figure 27] Prediction vs Ground Truth")

    sub = df[
        (df["architecture"] == "parallel") &
        (df["seed"] == 42) &
        (df["dataset"] == "pems-bay") &
        (df["horizon"] == 12)
    ]

    if sub.empty:
        print("  [warn] No PEMS-BAY H=12 data found. Skipping Figure 27.")
        return

    # Find best config
    best_row = sub.loc[sub["MAE"].idxmin()]

    # Build expected run folder name
    try:
        from Fusion_ST_variants.utils.io import run_name as _run_name
        rn = _run_name(
            dataset=str(best_row.get("dataset", "pems-bay")),
            horizon=int(best_row.get("horizon", 12)),
            gcn=str(best_row.get("gcn_variant", "")),
            tcn=str(best_row.get("tcn_variant", "")),
            arch=str(best_row.get("architecture", "parallel")),
            fusion=str(best_row.get("fusion_method", "")),
            seed=int(best_row.get("seed", 42)),
            parallel_spatial_mode="all_meanpool",
        )
        preds_path = RUNS_ROOT / rn / "predictions.npz"
    except Exception as e:
        print(f"  [warn] Could not determine predictions path: {e}")
        return

    if not preds_path.exists():
        print(f"  [warn] predictions.npz not found at {preds_path}")
        print("         Ensure Stage 2 Pass 2b ran with save_preds=True.")
        return

    data = np.load(str(preds_path))
    y_true = data["y_true"]   # (T, N, horizon, 1) or (T, N)
    y_pred = data["y_pred"]

    # Take node 0, first 200 time steps, H=1 step ahead
    node = 0
    n_show = min(200, y_true.shape[0])

    if y_true.ndim == 4:
        gt   = y_true[:n_show, node, 0, 0]
        pred = y_pred[:n_show, node, 0, 0]
    else:
        gt   = y_true[:n_show, node]
        pred = y_pred[:n_show, node]

    fig, ax = plt.subplots(figsize=(10.5, 3.8))
    ax.plot(gt,   color="#1565c0", linewidth=1.0, label="Ground Truth", alpha=0.9)
    ax.plot(pred, color="#c62828", linewidth=1.0, linestyle="--", label="Prediction", alpha=0.85)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Speed (mph)")
    ax.set_title(f"PEMS-BAY H=12 — {best_row.get('gcn_variant','')}+"
                 f"{best_row.get('tcn_variant','')} (MAE={float(best_row.get('MAE',0)):.3f})")
    ax.legend(fontsize=9)
    ax.grid(linestyle=":", alpha=0.5)
    plt.tight_layout()
    _savefig(fig, "fig27_pred_vs_gt")


# ══════════════════════════════════════════════════════════════════════════════
# 15. SUMMARY PRINT
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(df: pd.DataFrame) -> None:
    """Print a quick overview of best results per condition."""
    print("\n" + "=" * 70)
    print("BEST RESULTS SUMMARY (seed=42, parallel arch)")
    print("=" * 70)
    sub = df[(df["architecture"] == "parallel") & (df["seed"] == 42)]
    for ds in DATASETS:
        for h in HORIZONS:
            cond = sub[(sub["dataset"] == ds) & (sub["horizon"] == h)]
            if cond.empty:
                print(f"  {ds} H={h:2d}  — no data")
                continue
            best = cond.loc[cond["MAE"].idxmin()]
            print(
                f"  {ds} H={h:2d}  MAE={float(best['MAE']):.3f}  "
                f"RMSE={float(best['RMSE']):.3f}  "
                f"MAPE={float(best.get('MAPE', float('nan'))):.2f}%  "
                f"→  {best.get('gcn_variant','')}+{best.get('tcn_variant','')} "
                f"fusion={best.get('fusion_method','')}"
            )
    print("=" * 70)


# ══════════════════════════════════════════════════════════════════════════════
# 16. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 70)
    print("SERVER STAGE 5: Post-Training Analysis & Export")
    print(f"  Runs root : {RUNS_ROOT}")
    print(f"  Output    : {OUT_ROOT}")
    print("=" * 70)

    # ── Load ──────────────────────────────────────────────────────────────────
    df = load_results()
    print_summary(df)

    # ── Tables ────────────────────────────────────────────────────────────────
    print("\n──────────────── TABLES ────────────────")
    table_gcn(df)
    table_tcn(df)
    table_architecture(df)
    table_fusion(df)
    table_training_cost(df)
    table_model_complexity(df)
    table_statistical_robustness(df)

    # ── Figures ───────────────────────────────────────────────────────────────
    print("\n──────────────── FIGURES ───────────────")
    fig_gcn(df)
    fig_tcn(df)
    fig_fusion(df)
    fig_fgm_fam(df)
    fig_parallel_stacked(df)
    fig_perstep(df)
    fig_accuracy_vs_cost(df)
    fig_pred_vs_gt(df)

    print("\n" + "=" * 70)
    print("Analysis complete.")
    print(f"  Tables  → {OUT_TABLES}")
    print(f"  Figures → {OUT_FIGURES}")
    print("=" * 70)


if __name__ == "__main__":
    main()
