"""Metric utilities for traffic forecasting.

All metrics are computed on *denormalized* values.

Conventions:
- Model output/targets are expected as numpy arrays shaped (B, N, H, 1)
- `mean`/`std` are per-node arrays shaped (N,)

MAPE notes:
- MAPE can explode for very small denominators.
- We mask out |y_true| < `min_speed` (default 5.0) and report support percentage.
"""

from typing import Dict, Iterable, Tuple

import numpy as np


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    """Symmetric MAPE (percentage)."""
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    return float(np.mean(np.abs(y_true - y_pred) / (denom + eps)) * 100.0)


def _mape(y_true: np.ndarray, y_pred: np.ndarray, min_speed: float = 5.0, eps: float = 1e-6) -> Tuple[float, float]:
    """MAPE (percentage) with optional low-speed masking.

    Returns:
    - mape: float
    - support_pct: percent of samples included after masking
    """
    mask = np.abs(y_true) >= float(min_speed)
    if np.any(mask):
        val = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / (y_true[mask] + eps))) * 100.0)
        support = float(np.mean(mask) * 100.0)
        return val, support
    val = float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100.0)
    return val, 0.0


def denorm(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Inverse Z-score for arrays shaped (B, N, H, 1)."""
    return (x * std[None, :, None, None] + mean[None, :, None, None]).astype(np.float32)


def compute_metrics_denorm(y_true_norm: np.ndarray, y_pred_norm: np.ndarray, mean: np.ndarray, std: np.ndarray, steps: Iterable[int] = (3, 6, 12)) -> Dict[str, float]:
    """Compute aggregate and per-horizon-step metrics after inverse scaling."""
    yt = denorm(y_true_norm, mean, std)
    yp = denorm(y_pred_norm, mean, std)

    out: Dict[str, float] = {}

    yta = yt.reshape(-1)
    ypa = yp.reshape(-1)

    out["MAE"] = _mae(yta, ypa)
    out["RMSE"] = _rmse(yta, ypa)
    out["sMAPE"] = _smape(yta, ypa)
    mape, support = _mape(yta, ypa)
    out["MAPE"] = mape
    out["MAPE_support_pct"] = support

    horizon = int(yt.shape[2])
    for s in steps:
        i = int(s) - 1
        if i < 0 or i >= horizon:
            continue
        yts = yt[:, :, i : i + 1, :].reshape(-1)
        yps = yp[:, :, i : i + 1, :].reshape(-1)
        out[f"MAE@{s}"] = _mae(yts, yps)
        out[f"RMSE@{s}"] = _rmse(yts, yps)
        ms, _ = _mape(yts, yps)
        out[f"MAPE@{s}"] = ms

    return out
