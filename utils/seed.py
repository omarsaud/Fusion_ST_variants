"""Reproducibility and performance helpers."""

import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: Optional[int]) -> None:
    """Seed Python, NumPy and PyTorch RNGs."""
    if seed is None:
        return
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def enable_gpu_optimizations() -> None:
    """Enable TF32 + cuDNN benchmark for faster training on NVIDIA GPUs."""
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True


def is_windows() -> bool:
    """Return True if running on Windows."""
    return os.name == "nt"
