"""Fusion_ST_variants ablation runner.

This script is intended to be the single entry point you edit/run for ablations.

Protocol (DCRNN-style):
- Load one raw speed matrix (T, N)
- Chronological split by index ranges (70/20/10 by default)
- Fit Z-score scaler on the *training time slice only* (per-node mean/std)
- Train models on normalized data
- Evaluate and report metrics after inverse-scaling predictions/targets

Key shapes:
- Cached speed: (T, N) float32
- Dataset sample:
  - x: (N, history, 1)
  - y: (N, horizon, 1)
- Model output:
  - y_hat: (B, N, horizon, 1)

Outputs:
- Per-run folder under `Fusion_ST_variants_runs/<run_name>/`:
  - `config.json`, `best.pt`, `metrics.json`
- Consolidated CSV per dataset:
  - `Fusion_ST_variants_runs/results__<dataset>.csv`

Resume:
- If `metrics.json` exists for a run, it is skipped unless `--force` is used.
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import json
import time
from contextlib import nullcontext
from dataclasses import asdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from Fusion_ST_variants.models.forecaster import ForecasterConfig, GcnTcnForecaster
from Fusion_ST_variants.preprocessing.cache import cache_exists, load_cache, write_cache
from Fusion_ST_variants.preprocessing.load_raw import dataset_dirs, load_raw
from Fusion_ST_variants.preprocessing.scaler import compute_time_boundaries, fit_zscore
from Fusion_ST_variants.preprocessing.splits import make_split_starts
from Fusion_ST_variants.preprocessing.window_dataset import WindowDataset
from Fusion_ST_variants.utils.io import append_csv, ensure_dir, run_name, save_json
from Fusion_ST_variants.utils.metrics import compute_metrics_denorm
from Fusion_ST_variants.utils.seed import enable_gpu_optimizations, set_seed


DATASET = "[pems-bay]"   # "[metr-la, pems-bay, pems08]" 
HISTORY = 12
HORIZONS = [3]  # full set: [3, 6, 12]

TRAIN_RATIO = 0.7
VAL_RATIO = 0.2

GCN_VARIANTS = ["GWN"]  # full set: ["GCN", "DiffusionGCN", "GL", "SAGE", "GWN", "GCN+AM", "GCN+Gating", "LightGAT"]

TCN_VARIANTS = ["AMs"]  # full set: ["Default", "EnStr", "GMs", "AMs"]

ARCHITECTURES = ["parallel", "stack"]  # full set: ["parallel", "stack"]

FUSION_METHODS =  ["fgm","direct", "pmf", "fam"]  # full set: ["fgm", "direct", "pmf", "fam"]

SEEDS = [42]  # full set: [42]

BATCH_SIZE = 64
EPOCHS = 40
PATIENCE = 10
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
DROPOUT = 0.2

GCN_DIM = 32
TCN_DIM = 64

AMP = True

LOG_VAL_METRICS = True
SAVE_PREDS = False

SLEEP_BETWEEN_RUNS_S = 2.0

NUM_WORKERS = 0

PARALLEL_SPATIAL_MODE = "all_meanpool"  # choices: "last" (fast; uses x[..., -1, :]) | "all_meanpool" (fair; GCN over all timesteps then mean-pool)


def _parse_str_list(s: Optional[str]) -> Optional[List[str]]:
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None

    # Allow inputs like: "[metr-la, pems-bay, pems08]" or "(12, 6, 3)"
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
        s = s[1:-1].strip()

    parts = [p.strip().strip("\"").strip("'") for p in s.split(",")]
    parts = [p for p in parts if p]
    return parts or None


def _parse_int_list(s: Optional[str]) -> Optional[List[int]]:
    parts = _parse_str_list(s)
    if parts is None:
        return None
    return [int(p) for p in parts]


def _parse_parallel_spatial_mode(s: str) -> str:
    return str(s).strip().lower()


def _canonical_dataset_key(dataset_key: str) -> str:
    k = str(dataset_key).strip().lower()
    if k in {"metr-la", "metr_la", "metrla"}:
        return "metr-la"
    if k in {"pems-bay", "pems_bay", "pemsbay"}:
        return "pems-bay"
    if k in {"pems08", "pems-08", "pems_08"}:
        return "pems08"
    return k


def _parse_dataset_keys(dataset_arg: str) -> List[str]:
    s = str(dataset_arg).strip().lower()
    if not s:
        return [_canonical_dataset_key(DATASET)]

    if s in {"all", "all3", "three", "3"}:
        return ["metr-la", "pems-bay", "pems08"]

    parts = _parse_str_list(s)
    if parts is None:
        return [_canonical_dataset_key(s)]

    keys = [_canonical_dataset_key(p) for p in parts]
    seen: set[str] = set()
    out: List[str] = []
    for k in keys:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


def _project_root() -> Path:
    """Return repository root as a Path.

    The repo root is assumed to be the parent directory of the package folder.
    """
    return Path(__file__).resolve().parents[1]


def _package_root() -> Path:
    """Return the Fusion_ST_variants package directory as a Path."""
    return Path(__file__).resolve().parent


def _runs_root(project_root: Path) -> Path:
    """Return the directory where run artifacts and CSV reports are written."""
    return project_root / "Fusion_ST_variants_runs"


def _prepare_cached_dataset(
    dataset_key: str,
    train_ratio: float,
    val_ratio: float,
) -> Tuple[np.memmap, np.ndarray, np.ndarray, np.ndarray]:
    """Load (or build) cached normalized speed + scaler stats + adjacency.

    Returns:
    - speed_scaled: memmap view of cached (T, N) speed Z-scores
    - mean: (N,) train mean
    - std: (N,) train std
    - adj: (N, N) adjacency matrix
    """
    pkg_root = _package_root()
    if cache_exists(pkg_root, dataset_key):
        return load_cache(pkg_root, dataset_key)

    project_root = _project_root()
    data_dir = dataset_dirs(project_root)[dataset_key]
    speed, adj = load_raw(data_dir, dataset_key)

    train_end, _ = compute_time_boundaries(speed.shape[0], train_ratio=train_ratio, val_ratio=val_ratio)
    scaler = fit_zscore(speed, train_end_t=train_end)
    speed_scaled = scaler.transform(speed)

    write_cache(pkg_root, dataset_key, speed_scaled=speed_scaled, scaler=scaler, adj=adj)
    return load_cache(pkg_root, dataset_key)


def _make_loaders(
    speed_scaled: np.memmap,
    history: int,
    horizon: int,
    batch_size: int,
    train_ratio: float,
    val_ratio: float,
    device: torch.device,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any]]:
    """Create DataLoaders for train/val/test using index-range splits.

    The split is chronological, determined by whether the prediction window end
    falls before train_end / val_end.
    """
    splits = make_split_starts(speed_scaled.shape[0], history=history, horizon=horizon, train_ratio=train_ratio, val_ratio=val_ratio)

    train_ds = WindowDataset(speed_scaled, splits["train_idx"], history=history, horizon=horizon)
    val_ds = WindowDataset(speed_scaled, splits["val_idx"], history=history, horizon=horizon)
    test_ds = WindowDataset(speed_scaled, splits["test_idx"], history=history, horizon=horizon)

    pin = device.type == "cuda"
    nw = int(num_workers)

    train_loader = DataLoader(train_ds, batch_size=int(batch_size), shuffle=True, num_workers=nw, pin_memory=pin, drop_last=False, persistent_workers=(nw > 0))
    val_loader = DataLoader(val_ds, batch_size=int(batch_size), shuffle=False, num_workers=nw, pin_memory=pin, drop_last=False, persistent_workers=(nw > 0))
    test_loader = DataLoader(test_ds, batch_size=int(batch_size), shuffle=False, num_workers=nw, pin_memory=pin, drop_last=False, persistent_workers=(nw > 0))

    meta = {
        "train_end": int(splits["train_end"]),
        "val_end": int(splits["val_end"]),
        "num_train": int(len(train_ds)),
        "num_val": int(len(val_ds)),
        "num_test": int(len(test_ds)),
    }

    return train_loader, val_loader, test_loader, meta


def _eval_loss(
    model: nn.Module,
    loader: DataLoader,
    adj_t: torch.Tensor,
    device: torch.device,
    criterion: nn.Module,
    max_batches: int = 0,
) -> float:
    """Compute average loss on a loader (optionally only for first N batches)."""
    model.eval()
    losses: List[float] = []
    with torch.no_grad():
        for i, (xb, yb) in enumerate(loader):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            pred = model(xb, adj_t)
            losses.append(float(criterion(pred, yb).item()))
            if int(max_batches) > 0 and (i + 1) >= int(max_batches):
                break
    return float(np.mean(losses)) if losses else float("nan")


def _predict(model: nn.Module, loader: DataLoader, adj_t: torch.Tensor, device: torch.device, max_batches: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Collect y_true/y_pred arrays from a loader (optionally truncated).

    Returns arrays with shapes:
    - y_true_norm: (B_total, N, H, 1)
    - y_pred_norm: (B_total, N, H, 1)
    """
    model.eval()
    ys: List[np.ndarray] = []
    ps: List[np.ndarray] = []
    with torch.no_grad():
        for i, (xb, yb) in enumerate(loader):
            xb = xb.to(device, non_blocking=True)
            pred = model(xb, adj_t)
            ys.append(yb.numpy())
            ps.append(pred.detach().cpu().numpy())
            if int(max_batches) > 0 and (i + 1) >= int(max_batches):
                break
    return np.concatenate(ys, axis=0), np.concatenate(ps, axis=0)


def train_one_run(
    *,
    dataset_key: str,
    speed_scaled: np.memmap,
    mean: np.ndarray,
    std: np.ndarray,
    adj: np.ndarray,
    history: int,
    horizon: int,
    train_ratio: float,
    val_ratio: float,
    gcn: str,
    tcn: str,
    arch: str,
    fusion: str,
    seed: int,
    batch_size: int,
    epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
    dropout: float,
    gcn_dim: int,
    tcn_dim: int,
    amp: bool,
    results_csv: Path,
    runs_root: Path,
    force: bool,
    device: torch.device,
    max_train_batches: int,
    max_eval_batches: int,
    max_pred_batches: int,
    log_val_metrics: bool,
    save_preds: bool,
    parallel_spatial_mode: str,
    num_workers: int = 0,
    exp_i: int = 0,
    exp_n: int = 0,
) -> None:
    """Train/evaluate a single configuration and append results to CSV."""
    psm_eff = parallel_spatial_mode if str(arch).strip().lower() == "parallel" else "last"
    rn = run_name(
        dataset_key,
        horizon,
        gcn,
        tcn,
        arch,
        fusion,
        seed,
        parallel_spatial_mode=(psm_eff if str(arch).strip().lower() == "parallel" else None),
    )
    run_dir = runs_root / rn
    metrics_path = run_dir / "metrics.json"
    preds_path = run_dir / "preds_test.npz"

    if metrics_path.exists() and not force:
        if bool(save_preds) and (not preds_path.exists()):
            ensure_dir(run_dir)
            train_loader, val_loader, test_loader, _split_meta = _make_loaders(
                speed_scaled,
                history=history,
                horizon=horizon,
                batch_size=batch_size,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                device=device,
            )

            cfg_payload = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
            cfg = ForecasterConfig(**cfg_payload["model"])
            model = GcnTcnForecaster(cfg).to(device)
            best_path = run_dir / "best.pt"
            if best_path.exists():
                model.load_state_dict(torch.load(str(best_path), map_location=device, weights_only=True))

            adj_t = torch.from_numpy(adj).to(device=device, dtype=torch.float32)
            y_true_norm, y_pred_norm = _predict(
                model,
                test_loader,
                adj_t=adj_t,
                device=device,
                max_batches=int(max_pred_batches),
            )

            mean_b = mean.reshape(1, -1, 1, 1)
            std_b = std.reshape(1, -1, 1, 1)
            y_true = (y_true_norm * std_b + mean_b).astype(np.float32)
            y_pred = (y_pred_norm * std_b + mean_b).astype(np.float32)
            ensure_dir(preds_path.parent)
            np.savez_compressed(preds_path, y_true=y_true, y_pred=y_pred, mean=mean.astype(np.float32), std=std.astype(np.float32))
        return

    ensure_dir(run_dir)
    set_seed(seed)

    train_loader, val_loader, test_loader, split_meta = _make_loaders(
        speed_scaled,
        history=history,
        horizon=horizon,
        batch_size=batch_size,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        device=device,
        num_workers=num_workers,
    )

    cfg = ForecasterConfig(
        num_nodes=int(speed_scaled.shape[1]),
        history=history,
        horizon=horizon,
        gcn_variant=gcn,
        tcn_variant=tcn,
        architecture=arch,
        fusion_method=fusion,
        parallel_spatial_mode=str(psm_eff),
        gcn_dim=gcn_dim,
        tcn_dim=tcn_dim,
        dropout=dropout,
    )

    save_json(run_dir / "config.json", {"model": asdict(cfg), "seed": int(seed), "split": split_meta})

    model = GcnTcnForecaster(cfg).to(device)
    num_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    adj_t = torch.from_numpy(adj).to(device=device, dtype=torch.float32)

    criterion = nn.L1Loss()
    opt = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    use_amp = bool(amp) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    autocast_ctx = torch.amp.autocast(device_type="cuda", enabled=True) if use_amp else nullcontext()

    best_val = float("inf")
    best_epoch = 0
    bad = 0

    epoch_history: List[Dict[str, Any]] = []

    t0 = time.time()

    prefix = f"[exp {int(exp_i):03d}/{int(exp_n):03d}] " if int(exp_n) > 0 else ""
    print(f"{prefix}[run] {rn} device={device}")
    sys.stdout.flush()

    for ep in range(1, int(epochs) + 1):
        ep_t0 = time.time()
        model.train()
        train_losses: List[float] = []

        for i, (xb, yb) in enumerate(train_loader):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            if use_amp:
                with autocast_ctx:
                    pred = model(xb, adj_t)
                    loss = criterion(pred, yb)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(opt)
                scaler.update()
            else:
                pred = model(xb, adj_t)
                loss = criterion(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                opt.step()

            train_losses.append(float(loss.item()))

            if int(max_train_batches) > 0 and (i + 1) >= int(max_train_batches):
                break

        val_loss = _eval_loss(
            model,
            val_loader,
            adj_t=adj_t,
            device=device,
            criterion=criterion,
            max_batches=int(max_eval_batches),
        )

        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")

        is_best = False
        if val_loss < best_val:
            best_val = float(val_loss)
            best_epoch = int(ep)
            bad = 0
            best_pt = run_dir / "best.pt"
            ensure_dir(best_pt.parent)
            torch.save(model.state_dict(), str(best_pt))
            is_best = True
        else:
            bad += 1

        ep_time_s = float(time.time() - ep_t0)

        val_metrics: Optional[Dict[str, float]] = None
        if bool(log_val_metrics):
            yv_true_norm, yv_pred_norm = _predict(
                model,
                val_loader,
                adj_t=adj_t,
                device=device,
                max_batches=int(max_eval_batches),
            )
            val_metrics = compute_metrics_denorm(yv_true_norm, yv_pred_norm, mean=mean, std=std, steps=(3, 6, 12))
            print(
                f"[epoch {ep:03d}] time={ep_time_s:.1f}s "
                f"train={train_loss:.6f} val={float(val_loss):.6f} "
                f"MAE={val_metrics['MAE']:.3f} RMSE={val_metrics['RMSE']:.3f} "
                f"MAPE={val_metrics['MAPE']:.2f} sMAPE={val_metrics['sMAPE']:.2f} "
                f"best_val={best_val:.6f} bad={bad}{' *' if is_best else ''}"
            )
        else:
            print(
                f"[epoch {ep:03d}] time={ep_time_s:.1f}s "
                f"train={train_loss:.6f} val={float(val_loss):.6f} "
                f"best_val={best_val:.6f} bad={bad}{' *' if is_best else ''}"
            )

        epoch_row: Dict[str, Any] = {
            "epoch": int(ep),
            "epoch_time_s": float(ep_time_s),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "best_val_loss": float(best_val),
            "best_epoch": int(best_epoch),
            "bad": int(bad),
            "is_best": bool(is_best),
        }
        if val_metrics is not None:
            for k in ["MAE", "RMSE", "MAPE", "MAPE_support_pct", "sMAPE", "MAE@3", "RMSE@3", "MAPE@3", "MAE@6", "RMSE@6", "MAPE@6", "MAE@12", "RMSE@12", "MAPE@12"]:
                if k in val_metrics:
                    epoch_row[f"val_{k}"] = float(val_metrics[k])

        epoch_history.append(epoch_row)
        save_json(run_dir / "epoch_history.json", {"epochs": epoch_history})

        sys.stdout.flush()

        if bad >= int(patience):
            break

    train_time_s = float(time.time() - t0)
    epochs_ran = int(epoch_history[-1]["epoch"]) if epoch_history else 0
    avg_epoch_time_s = float(np.mean([r["epoch_time_s"] for r in epoch_history])) if epoch_history else float("nan")

    best_path = run_dir / "best.pt"
    if best_path.exists():
        model.load_state_dict(torch.load(str(best_path), map_location=device, weights_only=True))

    val_loss = _eval_loss(
        model,
        val_loader,
        adj_t=adj_t,
        device=device,
        criterion=criterion,
        max_batches=int(max_eval_batches),
    )
    test_loss = _eval_loss(
        model,
        test_loader,
        adj_t=adj_t,
        device=device,
        criterion=criterion,
        max_batches=int(max_eval_batches),
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t_infer0 = time.time()
    y_true_norm, y_pred_norm = _predict(model, test_loader, adj_t=adj_t, device=device, max_batches=int(max_pred_batches))
    inference_time_s = float(time.time() - t_infer0)
    peak_vram_mb = float(torch.cuda.max_memory_allocated(device) / 1e6) if device.type == "cuda" else 0.0

    diff_norm = (y_pred_norm - y_true_norm).astype(np.float32)
    mae_norm = float(np.mean(np.abs(diff_norm)))
    rmse_norm = float(np.sqrt(np.mean(diff_norm * diff_norm)))

    mean_b = mean.reshape(1, -1, 1, 1)
    std_b = std.reshape(1, -1, 1, 1)
    y_true = (y_true_norm * std_b + mean_b).astype(np.float32)
    y_pred = (y_pred_norm * std_b + mean_b).astype(np.float32)

    # Per-node MAE/RMSE — shape (N,) — lightweight, always saved for spatial analysis figures
    node_mae  = np.mean(np.abs(y_true - y_pred),  axis=(0, 2, 3))  # (N,)
    node_rmse = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=(0, 2, 3)))  # (N,)
    node_metrics_path = run_dir / "node_metrics.npz"
    ensure_dir(node_metrics_path.parent)
    np.savez_compressed(node_metrics_path, mae=node_mae.astype(np.float32), rmse=node_rmse.astype(np.float32))

    if bool(save_preds):
        ensure_dir(preds_path.parent)
        np.savez_compressed(preds_path, y_true=y_true, y_pred=y_pred, mean=mean.astype(np.float32), std=std.astype(np.float32))

    metrics = compute_metrics_denorm(y_true_norm, y_pred_norm, mean=mean, std=std, steps=(3, 6, 12))

    out: Dict[str, Any] = {
        "run_name": rn,
        "num_params": num_params,
        "dataset": dataset_key,
        "history": int(history),
        "horizon": int(horizon),
        "gcn": gcn,
        "tcn": tcn,
        "arch": arch,
        "fusion": fusion,
        "parallel_spatial_mode": str(psm_eff),
        "seed": int(seed),
        "batch_size": int(batch_size),
        "epochs": int(epochs),
        "patience": int(patience),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "dropout": float(dropout),
        "gcn_dim": int(gcn_dim),
        "tcn_dim": int(tcn_dim),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val),
        "val_loss": float(val_loss),
        "test_loss": float(test_loss),
        "train_time_s": train_time_s,
        "epochs_ran": int(epochs_ran),
        "avg_epoch_time_s": float(avg_epoch_time_s),
        "MAE_norm": float(mae_norm),
        "RMSE_norm": float(rmse_norm),
        "inference_time_s": float(inference_time_s),
        "peak_vram_mb": float(peak_vram_mb),
        **split_meta,
        **metrics,
    }

    save_json(metrics_path, out)

    header = [
        "run_name",
        "num_params",
        "dataset",
        "history",
        "horizon",
        "gcn",
        "tcn",
        "arch",
        "fusion",
        "parallel_spatial_mode",
        "seed",
        "batch_size",
        "epochs",
        "patience",
        "lr",
        "weight_decay",
        "dropout",
        "gcn_dim",
        "tcn_dim",
        "best_epoch",
        "best_val_loss",
        "val_loss",
        "test_loss",
        "train_time_s",
        "epochs_ran",
        "avg_epoch_time_s",
        "train_end",
        "val_end",
        "num_train",
        "num_val",
        "num_test",
        "MAE_norm",
        "RMSE_norm",
        "inference_time_s",
        "peak_vram_mb",
        "MAE",
        "RMSE",
        "MAPE",
        "MAPE_support_pct",
        "sMAPE",
        "MAE@3",
        "RMSE@3",
        "MAPE@3",
        "MAE@6",
        "RMSE@6",
        "MAPE@6",
        "MAE@12",
        "RMSE@12",
        "MAPE@12",
    ]

    append_csv(results_csv, out, header=header)

    try:
        mae = float(out.get("MAE", float("nan")))
        rmse = float(out.get("RMSE", float("nan")))
        mape = float(out.get("MAPE", float("nan")))
        smape = float(out.get("sMAPE", float("nan")))
    except Exception:
        mae, rmse, mape, smape = float("nan"), float("nan"), float("nan"), float("nan")

    print(
        f"{prefix}[done] best_epoch={int(best_epoch)} best_val={float(best_val):.6f} "
        f"test_loss={float(test_loss):.6f} time={float(train_time_s):.1f}s "
        f"MAE={mae:.3f} RMSE={rmse:.3f} MAPE={mape:.2f} sMAPE={smape:.2f}"
    )
    print(f"{prefix}[saved] {metrics_path} | {results_csv}")
    sys.stdout.flush()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default=DATASET)
    p.add_argument("--history", type=int, default=HISTORY)
    p.add_argument("--horizons", type=str, default="")
    p.add_argument("--gcn_variants", type=str, default="")
    p.add_argument("--tcn_variants", type=str, default="")
    p.add_argument("--architectures", type=str, default="")
    p.add_argument("--fusion_methods", type=str, default="")
    p.add_argument("--seeds", type=str, default="")
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--patience", type=int, default=PATIENCE)
    p.add_argument("--lr", type=float, default=LEARNING_RATE)
    p.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    p.add_argument("--dropout", type=float, default=DROPOUT)
    p.add_argument("--gcn_dim", type=int, default=GCN_DIM)
    p.add_argument("--tcn_dim", type=int, default=TCN_DIM)
    p.add_argument(
        "--parallel_spatial_mode",
        type=_parse_parallel_spatial_mode,
        default=PARALLEL_SPATIAL_MODE,
        choices=["last", "all_meanpool"],
        help="For arch=parallel only: 'last' (fast; GCN on last timestep) or 'all_meanpool' (fair; GCN on all timesteps then mean-pool).",
    )
    p.add_argument("--max_train_batches", type=int, default=0)
    p.add_argument("--max_eval_batches", type=int, default=0)
    p.add_argument("--max_pred_batches", type=int, default=0)
    p.add_argument("--val_metrics", action="store_true")
    p.add_argument("--no_val_metrics", action="store_true")
    p.add_argument("--save_preds", action="store_true")
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--sleep_between_runs_s", type=float, default=SLEEP_BETWEEN_RUNS_S)
    p.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main() -> None:
    root = str(_project_root())
    if root not in sys.path:
        sys.path.append(root)

    args = _parse_args()

    enable_gpu_optimizations()

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")

    runs_root = _runs_root(_project_root())
    ensure_dir(runs_root)

    dataset_keys = _parse_dataset_keys(args.dataset)

    horizons = _parse_int_list(args.horizons) or HORIZONS
    gcn_variants = _parse_str_list(args.gcn_variants) or GCN_VARIANTS
    tcn_variants = _parse_str_list(args.tcn_variants) or TCN_VARIANTS
    archs = _parse_str_list(args.architectures) or ARCHITECTURES
    fusions = _parse_str_list(args.fusion_methods) or FUSION_METHODS
    seeds = _parse_int_list(args.seeds) or SEEDS

    history = int(args.history)
    batch_size = int(args.batch_size)
    epochs = int(args.epochs)
    patience = int(args.patience)
    lr = float(args.lr)
    weight_decay = float(args.weight_decay)
    dropout = float(args.dropout)
    gcn_dim = int(args.gcn_dim)
    tcn_dim = int(args.tcn_dim)
    amp = bool(AMP) and (not bool(args.no_amp))

    log_val_metrics = bool(LOG_VAL_METRICS)
    if bool(args.val_metrics):
        log_val_metrics = True
    if bool(args.no_val_metrics):
        log_val_metrics = False

    max_train_batches = int(args.max_train_batches)
    max_eval_batches = int(args.max_eval_batches)
    max_pred_batches = int(args.max_pred_batches)
    save_preds = bool(getattr(args, "save_preds", False)) or bool(SAVE_PREDS)

    sleep_between_runs_s = float(args.sleep_between_runs_s)
    num_workers = int(getattr(args, "num_workers", NUM_WORKERS))

    parallel_spatial_mode = str(getattr(args, "parallel_spatial_mode", PARALLEL_SPATIAL_MODE)).strip().lower() or PARALLEL_SPATIAL_MODE

    if bool(args.smoke):
        horizons = horizons[:1]
        gcn_variants = gcn_variants[:1]
        tcn_variants = tcn_variants[:1]
        archs = archs[:1]
        fusions = fusions[:1]
        seeds = seeds[:1]

        if max_train_batches <= 0:
            max_train_batches = 10
        if max_eval_batches <= 0:
            max_eval_batches = 10
        if max_pred_batches <= 0:
            max_pred_batches = 10

    force = bool(args.force)

    for dataset_key in dataset_keys:
        speed_scaled, mean, std, adj = _prepare_cached_dataset(dataset_key, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO)
        results_csv = runs_root / f"results__{dataset_key}.csv"

        grid: List[Tuple[int, str, str, str, str, int, str]] = []
        for horizon in horizons:
            for gcn in gcn_variants:
                for tcn in tcn_variants:
                    for arch in archs:
                        for fusion in fusions:
                            if str(arch).strip().lower() == "stack" and str(fusion).strip().lower() != "direct":
                                continue
                            for seed in seeds:
                                grid.append((int(horizon), str(gcn), str(tcn), str(arch), str(fusion), int(seed), str(parallel_spatial_mode)))

        to_run: List[Tuple[int, str, str, str, str, int, str]] = []
        skipped = 0
        for horizon, gcn, tcn, arch, fusion, seed, psm in grid:
            psm_eff = psm if str(arch).strip().lower() == "parallel" else "last"
            rn = run_name(
                dataset_key,
                horizon,
                gcn,
                tcn,
                arch,
                fusion,
                seed,
                parallel_spatial_mode=(psm_eff if str(arch).strip().lower() == "parallel" else None),
            )
            metrics_path = (runs_root / rn / "metrics.json")
            if metrics_path.exists() and (not force):
                skipped += 1
            else:
                to_run.append((horizon, gcn, tcn, arch, fusion, seed, psm))

        print(f"[dataset] {dataset_key} | planned={len(grid)} to_run={len(to_run)} skipped={skipped} force={force}")
        sys.stdout.flush()

        total = int(len(to_run))
        for i, (horizon, gcn, tcn, arch, fusion, seed, psm) in enumerate(to_run, start=1):
            remaining_after = int(total - i)
            bar = "=" * 110
            print("\n" + bar)
            print(
                f"[exp {int(i):03d}/{int(total):03d}] START remaining_after={remaining_after} "
                f"dataset={dataset_key} history={int(history)} horizon={int(horizon)} "
                f"gcn={gcn} tcn={tcn} arch={arch} fusion={fusion} psm={str(psm)} seed={int(seed)}"
            )
            print(bar)
            sys.stdout.flush()

            train_one_run(
                dataset_key=dataset_key,
                speed_scaled=speed_scaled,
                mean=mean,
                std=std,
                adj=adj,
                history=history,
                horizon=horizon,
                train_ratio=TRAIN_RATIO,
                val_ratio=VAL_RATIO,
                gcn=gcn,
                tcn=tcn,
                arch=arch,
                fusion=fusion,
                seed=seed,
                batch_size=batch_size,
                epochs=epochs,
                patience=patience,
                lr=lr,
                weight_decay=weight_decay,
                dropout=dropout,
                gcn_dim=gcn_dim,
                tcn_dim=tcn_dim,
                amp=amp,
                results_csv=results_csv,
                runs_root=runs_root,
                force=force,
                device=device,
                max_train_batches=max_train_batches,
                max_eval_batches=max_eval_batches,
                max_pred_batches=max_pred_batches,
                log_val_metrics=log_val_metrics,
                save_preds=save_preds,
                parallel_spatial_mode=psm,
                num_workers=num_workers,
                exp_i=i,
                exp_n=int(len(to_run)),
            )

            if sleep_between_runs_s > 0 and i < total:
                print(f"[pause] sleeping {float(sleep_between_runs_s):.1f}s before next scenario...")
                sys.stdout.flush()
                time.sleep(float(sleep_between_runs_s))


if __name__ == "__main__":
    main()
