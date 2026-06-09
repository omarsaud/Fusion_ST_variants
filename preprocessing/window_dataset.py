"""Windowed PyTorch dataset for speed forecasting.
 
 Input speed array:
 - speed_scaled: (T, N) float32 (often a memmap)
 
 For each start index `s`, we return:
 - x: (N, history, 1)
 - y: (N, horizon, 1)
 """
 
from typing import Union

import numpy as np
import torch
from torch.utils.data import Dataset


class WindowDataset(Dataset):
    def __init__(self, speed_scaled: Union[np.ndarray, np.memmap], starts: np.ndarray, history: int, horizon: int):
        """Create a dataset over precomputed start indices."""
        self.speed = speed_scaled
        self.starts = starts.astype(np.int64)
        self.history = int(history)
        self.horizon = int(horizon)

    def __len__(self) -> int:
        return int(self.starts.shape[0])

    def __getitem__(self, idx: int):
        """Return one (x, y) sample as torch float tensors."""
        s = int(self.starts[int(idx)])
        x = self.speed[s : s + self.history]
        y = self.speed[s + self.history : s + self.history + self.horizon]

        x = x[:, :, None]
        y = y[:, :, None]

        x = np.transpose(x, (1, 0, 2))
        y = np.transpose(y, (1, 0, 2))

        return torch.from_numpy(np.ascontiguousarray(x)).float(), torch.from_numpy(np.ascontiguousarray(y)).float()
