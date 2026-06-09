"""Scaling utilities (train-only Z-score).

The scaler is fit on the *training time slice only* and computed per node:
- speed: (T, N)
- mean/std: (N,)

This matches benchmark-safe practice (no leakage from val/test into scaler).

Boundary computation utilities for splitting time series into training, validation, and test sets.
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class ZScoreScaler:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Apply z-score: (x - mean) / std."""
        return ((x - self.mean) / self.std).astype(np.float32)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        """Inverse z-score: x * std + mean."""
        return (x * self.std + self.mean).astype(np.float32)


def fit_zscore(speed: np.ndarray, train_end_t: int) -> ZScoreScaler:
    """Fit per-node mean/std using `speed[:train_end_t]` only."""
    train_slice = speed[: int(train_end_t)]
    mean = train_slice.mean(axis=0, dtype=np.float64)
    std = train_slice.std(axis=0, dtype=np.float64)
    std = np.where(std < 1e-6, 1.0, std)
    return ZScoreScaler(mean=mean.astype(np.float32), std=std.astype(np.float32))


def compute_time_boundaries(num_timesteps: int, train_ratio: float, val_ratio: float) -> Tuple[int, int]:
    """Compute train_end and val_end timestep boundaries.

    Returns:
    - train_end: int
    - val_end: int
    """
    train_end = int(num_timesteps * float(train_ratio))
    val_end = int(num_timesteps * float(train_ratio + val_ratio))
    train_end = max(1, min(train_end, num_timesteps))
    val_end = max(train_end + 1, min(val_end, num_timesteps))
    return train_end, val_end
