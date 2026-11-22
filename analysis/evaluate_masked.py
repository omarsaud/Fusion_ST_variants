"""
Masked re-evaluation of trained runs on the held-out test set (inference only).

METR-LA and PEMS-BAY store missing sensor readings as 0 mph. The headline MAE
and RMSE written by ``train_ablation.py`` average the error over every test
reading, including these zeros. This script reloads each run's ``best.pt``,
predicts the test set once (no training, no weight update), and reports the
error with missing readings excluded:

- missing reading : true speed < ``--null`` mph (default 0.5; the stored zeros
                    come back as ~1e-5 after z-score inversion)
- MAE/RMSE_masked : over all forecast steps 1..Q, missing readings excluded
- *_stepQ         : the final forecast step Q only, missing readings excluded
- congested / free: valid readings below / at or above ``--congested`` mph
                    (default 40)

Consistency check: the script also recomputes the unmasked MAE over all steps
and compares it with the MAE stored in the run's ``metrics.json``
(``abs_diff`` column; ``match`` is True when the difference is below
``--tol``). Masked metrics are written for every run that loads.

Each run directory must contain ``config.json``, ``metrics.json`` and
``best.pt`` as written by ``train_ablation.py``. The dataset cache is reused
(or built from the raw data) exactly as during training.

Usage:
    python -m Fusion_ST_variants.analysis.evaluate_masked                  # every run under Fusion_ST_variants_runs/
    python -m Fusion_ST_variants.analysis.evaluate_masked RUN_DIR [RUN_DIR ...] --out masked_eval.csv
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from Fusion_ST_variants import train_ablation as ta
from Fusion_ST_variants.models.forecaster import ForecasterConfig, GcnTcnForecaster
from Fusion_ST_variants.preprocessing.splits import make_split_starts
from Fusion_ST_variants.preprocessing.window_dataset import WindowDataset

FIELDS = [
    "run_name", "dataset", "Q", "gcn", "tcn", "arch", "fusion", "seed", "num_params",
    "logged_MAE", "recomputed_MAE", "abs_diff", "match",
    "MAE_masked", "RMSE_masked", "MAE_masked_stepQ", "RMSE_masked_stepQ", "missing_pct",
    "MAE_masked_congested", "MAE_masked_free", "congested_pct",
]


def _mae_rmse(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray):
    err = y_true[mask] - y_pred[mask]
    return float(np.abs(err).mean()), float(np.sqrt((err ** 2).mean()))


def _find_runs(runs_root: Path):
    return sorted(p for p in runs_root.iterdir()
                  if p.is_dir() and all((p / f).exists() for f in ("config.json", "metrics.json", "best.pt")))


def evaluate_run(run_dir: Path, data_cache: dict, device: torch.device, null: float, congested: float, tol: float) -> dict:
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    logged = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    ds = logged["dataset"]
    if ds not in data_cache:
        speed, mean, std, adj = ta._prepare_cached_dataset(ds, train_ratio=ta.TRAIN_RATIO, val_ratio=ta.VAL_RATIO)
        data_cache[ds] = (speed, mean, std, torch.from_numpy(adj).to(device, dtype=torch.float32))
    speed, mean, std, adj_t = data_cache[ds]

    Q, hist = int(cfg["model"]["horizon"]), int(cfg["model"]["history"])
    sp = make_split_starts(speed.shape[0], history=hist, horizon=Q, train_ratio=ta.TRAIN_RATIO, val_ratio=ta.VAL_RATIO)
    if int(sp["train_end"]) != cfg["split"]["train_end"] or len(sp["test_idx"]) != cfg["split"]["num_test"]:
        raise ValueError(f"test split differs from the one recorded in {run_dir / 'config.json'}")
    loader = DataLoader(WindowDataset(speed, sp["test_idx"], history=hist, horizon=Q), batch_size=128, shuffle=False)

    model = GcnTcnForecaster(ForecasterConfig(**cfg["model"])).to(device)
    model.load_state_dict(torch.load(str(run_dir / "best.pt"), map_location=device, weights_only=True))
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for xb, yb in loader:
            ps.append(model(xb.to(device), adj_t).cpu().numpy())
            ys.append(yb.numpy())
    m, s = mean.reshape(1, -1, 1, 1), std.reshape(1, -1, 1, 1)
    y_true = np.concatenate(ys) * s + m          # (T, N, Q, 1), mph
    y_pred = np.concatenate(ps) * s + m

    rec = float(np.abs(y_true - y_pred).mean())
    valid = y_true >= null
    cong, free = valid & (y_true < congested), valid & (y_true >= congested)
    row = dict(run_name=run_dir.name, dataset=ds, Q=Q, gcn=logged["gcn"], tcn=logged["tcn"], arch=logged["arch"],
               fusion=logged["fusion"], seed=logged["seed"], num_params=logged.get("num_params", ""),
               logged_MAE=round(logged["MAE"], 6), recomputed_MAE=round(rec, 6),
               abs_diff=round(abs(rec - logged["MAE"]), 6), match=abs(rec - logged["MAE"]) < tol,
               missing_pct=round(100 * (1 - valid.mean()), 3),
               congested_pct=round(100 * cong.sum() / valid.sum(), 3))
    row["MAE_masked"], row["RMSE_masked"] = _mae_rmse(y_true, y_pred, valid)
    last = (slice(None), slice(None), slice(Q - 1, Q), slice(None))
    row["MAE_masked_stepQ"], row["RMSE_masked_stepQ"] = _mae_rmse(y_true[last], y_pred[last], valid[last])
    row["MAE_masked_congested"] = _mae_rmse(y_true, y_pred, cong)[0]
    row["MAE_masked_free"] = _mae_rmse(y_true, y_pred, free)[0]
    return row


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dirs", nargs="*", type=Path, help="run directories (default: every run under --runs_root)")
    p.add_argument("--runs_root", type=Path, default=ta._runs_root(ta._project_root()))
    p.add_argument("--out", type=Path, default=Path("masked_eval.csv"))
    p.add_argument("--null", type=float, default=0.5, help="readings below this speed (mph) are treated as missing")
    p.add_argument("--congested", type=float, default=40.0, help="congestion threshold (mph)")
    p.add_argument("--tol", type=float, default=1e-3, help="tolerance for the logged-MAE consistency check")
    args = p.parse_args()

    runs = args.run_dirs or _find_runs(args.runs_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_cache, rows = {}, []
    for k, rd in enumerate(runs, 1):
        try:
            row = evaluate_run(rd, data_cache, device, args.null, args.congested, args.tol)
        except Exception as exc:  # report and continue with the next run
            print(f"[{k}/{len(runs)}] ERROR {rd.name}: {exc}", flush=True)
            continue
        rows.append(row)
        print(f"[{k}/{len(runs)}] {rd.name}  MAE logged={row['logged_MAE']:.4f} recomputed={row['recomputed_MAE']:.4f}"
              f"  masked={row['MAE_masked']:.4f} stepQ={row['MAE_masked_stepQ']:.4f}", flush=True)

    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} of {len(runs)} runs evaluated, {sum(r['match'] for r in rows)} within tol={args.tol} -> {args.out}")


if __name__ == "__main__":
    main()
