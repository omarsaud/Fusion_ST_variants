"""Chronological train/val/test split by window start indices.
 
 Given a full time series (T, ...), we create all possible window start indices
 `s` such that `s + history + horizon <= T`.
 
 Each start index is assigned to train/val/test based on the *end* of its
 prediction window:
 - train: pred_end < train_end
 - val:   train_end <= pred_end < val_end
 - test:  pred_end >= val_end
 """
 
from typing import Dict

import numpy as np


def make_split_starts(num_timesteps: int, history: int, horizon: int, train_ratio: float, val_ratio: float) -> Dict[str, np.ndarray]:
    """Return start indices for each split and the time boundaries.

    Returns a dict with keys:
    - train_end: scalar int64
    - val_end: scalar int64
    - train_idx/val_idx/test_idx: 1D int64 arrays of start indices
    """
    train_end = int(num_timesteps * float(train_ratio))
    val_end = int(num_timesteps * float(train_ratio + val_ratio))

    max_start = int(num_timesteps) - int(history) - int(horizon)
    if max_start < 0:
        raise ValueError("Not enough timesteps for the requested history/horizon")

    starts = np.arange(max_start + 1, dtype=np.int64)
    pred_end = starts + int(history) + int(horizon) - 1

    train_idx = starts[pred_end < train_end]
    val_idx = starts[(pred_end >= train_end) & (pred_end < val_end)]
    test_idx = starts[pred_end >= val_end]

    return {
        "train_end": np.asarray(train_end, dtype=np.int64),
        "val_end": np.asarray(val_end, dtype=np.int64),
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
    }
