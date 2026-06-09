"""Disk cache for preprocessed artifacts.
 
 Cached per dataset:
 - speed_scaled.npy : (T, N) float32 (loaded as memmap for efficiency)
 - mean.npy, std.npy: (N,) float32
 - adj.npy          : (N, N) float32
 """
 
from pathlib import Path
from typing import Tuple
 
import numpy as np
 
from Fusion_ST_variants.preprocessing.scaler import ZScoreScaler


def cache_dir(package_root: Path) -> Path:
    """Root cache directory under the package folder."""
    return package_root / "cache"


def dataset_cache_dir(package_root: Path, dataset_key: str) -> Path:
    """Cache directory for a specific dataset key."""
    return cache_dir(package_root) / dataset_key


def cache_exists(package_root: Path, dataset_key: str) -> bool:
    """Return True if all required cached artifacts exist."""
    d = dataset_cache_dir(package_root, dataset_key)
    return (d / "speed_scaled.npy").exists() and (d / "mean.npy").exists() and (d / "std.npy").exists() and (d / "adj.npy").exists()


def load_cache(package_root: Path, dataset_key: str) -> Tuple[np.memmap, np.ndarray, np.ndarray, np.ndarray]:
    """Load cached artifacts; speed is loaded as a memmap for low RAM usage."""
    d = dataset_cache_dir(package_root, dataset_key)
    speed = np.load(str(d / "speed_scaled.npy"), mmap_mode="r")
    mean = np.load(str(d / "mean.npy")).astype(np.float32)
    std = np.load(str(d / "std.npy")).astype(np.float32)
    adj = np.load(str(d / "adj.npy")).astype(np.float32)
    return speed, mean, std, adj


def write_cache(package_root: Path, dataset_key: str, speed_scaled: np.ndarray, scaler: ZScoreScaler, adj: np.ndarray) -> None:
    """Write cached artifacts to disk."""
    d = dataset_cache_dir(package_root, dataset_key)
    d.mkdir(parents=True, exist_ok=True)
    np.save(str(d / "speed_scaled.npy"), speed_scaled.astype(np.float32))
    np.save(str(d / "mean.npy"), scaler.mean.astype(np.float32))
    np.save(str(d / "std.npy"), scaler.std.astype(np.float32))
    np.save(str(d / "adj.npy"), adj.astype(np.float32))
