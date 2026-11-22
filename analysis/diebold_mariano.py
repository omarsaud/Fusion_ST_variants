"""
Diebold-Mariano test of equal predictive accuracy for two trained runs.

Compares two forecasts on the held-out test set using their per-instance
absolute errors. For two forecasts with errors e1_t and e2_t over the T
time-ordered test instances (each instance's error averaged over all sensors
and the full Q-step forecast), the loss differential is

    d_t = |e1_t| - |e2_t|

and the statistic is

    DM = mean(d) / sqrt(var_nw(d)),

where var_nw is a Newey-West heteroskedasticity- and autocorrelation-consistent
variance estimator with a Bartlett kernel and bandwidth Q-1 for a Q-step
horizon. Under the null of equal predictive accuracy DM ~ N(0, 1); a negative
DM means the first forecast (A) is more accurate than the second (B).

Each run directory is expected to contain a ``preds_test.npz`` file with arrays
``y_true`` and ``y_pred`` of shape (T, N, Q, 1) on the denormalized scale.
These are written by ``train_ablation.py`` when run with ``--save_preds`` (the
evaluation pass loads the saved ``best.pt`` checkpoint; no retraining occurs and
the reported MAE is unchanged).

Usage:
    python -m Fusion_ST_variants.analysis.diebold_mariano RUN_A_DIR RUN_B_DIR --horizon 12
"""

import argparse
import math
from pathlib import Path

import numpy as np


def per_instance_error(preds_path: Path) -> np.ndarray:
    """Per-test-instance MAE averaged over sensors and forecast steps -> (T,)."""
    z = np.load(preds_path)
    abs_err = np.abs(z["y_pred"] - z["y_true"])      # (T, N, Q, 1)
    return abs_err.mean(axis=(1, 2, 3))               # (T,)


def aggregate_mae(preds_path: Path) -> float:
    z = np.load(preds_path)
    return float(np.abs(z["y_pred"] - z["y_true"]).mean())


def diebold_mariano(e_a: np.ndarray, e_b: np.ndarray, horizon: int):
    """Return (DM statistic, two-sided p-value). Negative DM favours A."""
    d = e_a - e_b
    n = len(d)
    d_bar = d.mean()
    dc = d - d_bar
    gamma0 = float(dc @ dc) / n
    var = gamma0
    for k in range(1, max(1, horizon)):
        if k >= n:
            break
        gamma_k = float(dc[k:] @ dc[:-k]) / n
        var += 2.0 * (1.0 - k / horizon) * gamma_k
    var /= n
    if var <= 0:
        return float("nan"), float("nan")
    dm = d_bar / math.sqrt(var)
    # two-sided normal p-value
    p = math.erfc(abs(dm) / math.sqrt(2.0))
    return dm, p


def main() -> None:
    ap = argparse.ArgumentParser(description="Diebold-Mariano test between two runs.")
    ap.add_argument("run_a", type=str, help="run directory A (contains preds_test.npz)")
    ap.add_argument("run_b", type=str, help="run directory B (contains preds_test.npz)")
    ap.add_argument("--horizon", type=int, required=True, help="forecast horizon Q (e.g. 3, 6, 12)")
    args = ap.parse_args()

    pa = Path(args.run_a) / "preds_test.npz"
    pb = Path(args.run_b) / "preds_test.npz"
    for p in (pa, pb):
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found. Generate it first with:\n"
                f"  python -m Fusion_ST_variants.train_ablation ... --save_preds"
            )

    e_a, e_b = per_instance_error(pa), per_instance_error(pb)
    dm, p = diebold_mariano(e_a, e_b, args.horizon)
    better = "A" if dm < 0 else "B"

    print(f"MAE(A) = {aggregate_mae(pa):.4f}   MAE(B) = {aggregate_mae(pb):.4f}")
    print(f"DM = {dm:.3f}   p = {p:.3e}   T = {len(e_a)}   more accurate: {better}")


if __name__ == "__main__":
    main()
