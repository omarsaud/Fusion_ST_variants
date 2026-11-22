"""Raw data loading utilities.

This module loads:
- Speed time series from dataset `.h5` files into a dense matrix of shape (T, N)
- The adjacency matrix from `adj_mx.pkl` into a dense matrix of shape (N, N)

Dataset-specific HDF5 keys:
- METR-LA: `df/block0_values`
- PEMS-BAY: `speed/block0_values`
"""

import os
import pickle
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np


def _read_speed_matrix(h5_path: Path, dataset_key: str) -> np.ndarray:
    """Read raw speed matrix from an HDF5 file (returns float array (T, N))."""
    with h5py.File(str(h5_path), "r") as f:
        key = dataset_key.strip().lower()
        if key in {"metr-la", "metr_la", "metrla"}:
            return f["df"]["block0_values"][:]
        if key in {"pems-bay", "pems_bay", "pemsbay"}:
            return f["speed"]["block0_values"][:]
        raise ValueError(f"Unknown dataset_key: {dataset_key}")


def _read_speed_matrix_npz(npz_path: Path) -> np.ndarray:
    """Read a raw speed matrix from a `.npz` file.

    Supports common formats:
    - data shape (T, N)
    - data shape (T, N, F) where one channel corresponds to speed.
    """
    obj = np.load(str(npz_path))
    if "data" not in obj:
        raise KeyError(f"Expected key 'data' in npz: {npz_path}")
    data = np.asarray(obj["data"]).astype(np.float32)

    if data.ndim == 2:
        return data

    if data.ndim != 3:
        raise ValueError(f"Unsupported npz data shape {tuple(data.shape)} in {npz_path}")

    t, n, f = data.shape
    if f == 1:
        return data[..., 0]

    # Heuristic to select a speed-like channel when the dataset is multi-feature.
    # Typical ranges:
    # - occupancy: ~[0, 1]
    # - speed: ~[0, 120]
    # - flow: can be much larger (often > 200)
    p95_per_f = [float(np.nanpercentile(data[..., i], 95)) for i in range(int(f))]
    max_per_f = [float(np.nanmax(data[..., i])) for i in range(int(f))]

    candidates: List[int] = []
    for i in range(int(f)):
        p95 = p95_per_f[i]
        mx = max_per_f[i]
        if (p95 > 2.0 and p95 < 150.0) and (mx > 2.0 and mx < 220.0):
            candidates.append(i)

    # Prefer the smallest-range channel among candidates (usually speed), else fallback.
    if candidates:
        idx = int(sorted(candidates, key=lambda i: p95_per_f[i])[0])
    else:
        idx = int([i for i, mx in enumerate(max_per_f) if (mx > 2.0 and mx < 200.0)][0]) if any((mx > 2.0 and mx < 200.0) for mx in max_per_f) else int(f - 1)

    return data[..., idx]


def _adj_from_distance_csv(distance_csv_path: Path, num_nodes: int) -> np.ndarray:
    """Build a weighted adjacency matrix from a distance CSV.

    Expected CSV columns (common in traffic repos): from, to, distance.
    """
    edges: List[Tuple[int, int, float]] = []
    dists: List[float] = []
    with open(str(distance_csv_path), "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"Empty distance csv: {distance_csv_path}")

        for row in reader:
            if not row:
                continue
            i = int(float(row[0]))
            j = int(float(row[1]))
            d = float(row[2])
            edges.append((i, j, d))
            dists.append(d)

    if not edges:
        raise ValueError(f"No edges found in: {distance_csv_path}")

    sigma = float(np.std(np.asarray(dists, dtype=np.float32)))
    if not np.isfinite(sigma) or sigma <= 0.0:
        sigma = 1.0

    adj = np.zeros((int(num_nodes), int(num_nodes)), dtype=np.float32)
    for i, j, d in edges:
        if 0 <= i < num_nodes and 0 <= j < num_nodes:
            w = float(np.exp(-((d / sigma) ** 2)))
            if w >= 0.1:
                adj[i, j] = max(adj[i, j], w)
                adj[j, i] = max(adj[j, i], w)

    np.fill_diagonal(adj, 1.0)
    adj = adj / (adj.sum(axis=1, keepdims=True) + 1e-6)
    return adj


def _read_adj_matrix(pkl_path: Path) -> np.ndarray:
    """Read adjacency matrix from a pickle file.

    Many traffic repos store a tuple/list `(sensor_ids, id_to_ind, adj)`.
    If so, we return the third element.
    """
    with open(str(pkl_path), "rb") as f:
        obj = pickle.load(f, encoding="latin1")
    if isinstance(obj, (list, tuple)) and len(obj) >= 3:
        return np.asarray(obj[2])
    return np.asarray(obj)


def load_raw(data_dir: Path, dataset_key: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load raw speed and adjacency for a dataset.

    Returns:
    - speed: (T, N) float32
    - adj: (N, N) float32
    """
    key = dataset_key.strip().lower()
    if key in {"metr-la", "metr_la", "metrla"}:
        h5_path = data_dir / "metr-la.h5"
    elif key in {"pems-bay", "pems_bay", "pemsbay"}:
        h5_path = data_dir / "pems-bay.h5"
    elif key in {"pems08", "pems-08", "pems_08"}:
        h5_path = data_dir / "pems08.h5"
    else:
        raise ValueError(f"Unknown dataset_key: {dataset_key}")

    if key in {"pems08", "pems-08", "pems_08"}:
        npz_candidates = [data_dir / "pems08.npz", data_dir / "PEMS08.npz"]
        npz_path = next((p for p in npz_candidates if p.exists()), None)
        if npz_path is not None:
            speed = _read_speed_matrix_npz(npz_path).astype(np.float32)
        elif h5_path.exists():
            speed = _read_speed_matrix(h5_path, dataset_key).astype(np.float32)
        else:
            raise FileNotFoundError(str(npz_candidates[0]))
    else:
        if not h5_path.exists():
            raise FileNotFoundError(str(h5_path))
        speed = _read_speed_matrix(h5_path, dataset_key).astype(np.float32)

    if np.isnan(speed).any():
        raise ValueError(f"NaNs found in speed matrix: {data_dir}")

    adj_pkl_candidates = [data_dir / "adj_mx.pkl", data_dir / "adj_pems08.pkl", data_dir / "adj_mx_bay.pkl"]
    adj_pkl = next((p for p in adj_pkl_candidates if p.exists()), None)
    if adj_pkl is not None:
        adj = _read_adj_matrix(adj_pkl).astype(np.float32)
    else:
        dist_csv_candidates = [
            data_dir / "distance.csv",
            data_dir / "distance08.csv",
            data_dir / "distance_pems08.csv",
            data_dir / "PEMS08.csv",
        ]
        dist_csv = next((p for p in dist_csv_candidates if p.exists()), None)
        if dist_csv is not None:
            adj = _adj_from_distance_csv(dist_csv, num_nodes=int(speed.shape[1]))
        else:
            raise FileNotFoundError(str(data_dir / "adj_mx.pkl"))

    return speed, adj


def dataset_dirs(project_root: Path) -> Dict[str, Path]:
    """Default dataset directory mapping under `<repo>/data/`."""
    return {
        "metr-la": project_root / "data" / "metr-la",
        "pems-bay": project_root / "data" / "pems-bay",
        "pems08": project_root / "data" / "pems08",
    }
