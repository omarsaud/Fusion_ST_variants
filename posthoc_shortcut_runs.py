"""
POST-HOC RUNS — shortcut controls and single-precision reruns (17 runs)
=======================================================================
These runs were added after the main 255-run comparison had been analysed.

Control encoders (registered here only; models/ is not modified). raw(x) is the encoder output before its final ReLU:
  <enc>+SkipOnly = ReLU(raw(x) + W x)
  <enc>+NormOnly = LayerNorm(ReLU(raw(x)))
  <enc>+Skip     = LayerNorm(ReLU(raw(x) + W x))
with <enc> in GCN, DiffusionGCN (dGCN), GWN (Spectral GCN).

Runs:
  14 control runs (D-TCN, H=12, tcn_dim 64, lr 1e-3, AMP) including two plain METR-LA GCN baselines
   3 reruns of the stacked Spectral GCN + TCN+AMs METR-LA runs (H=3, 6, 12) in single precision,
     tcn_dim 128, lr 5e-4; their original runs had a non-finite loss in every epoch.

Run from the project root (parent of this repository):
  python Fusion_ST_variants/posthoc_shortcut_runs.py --list
  python Fusion_ST_variants/posthoc_shortcut_runs.py --test            (CPU, no data needed)
  python Fusion_ST_variants/posthoc_shortcut_runs.py --data-root data --runs-root runs_posthoc [--only 0,3]
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for p in (str(_HERE), str(_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
import torch.nn as nn
import Fusion_ST_variants.models.forecaster as fc
from Fusion_ST_variants.models.gcn_variants import DiffusionGCN, GCNLayer, GraphWaveletNetwork

ENCODERS = {"GCN": GCNLayer, "DiffusionGCN": DiffusionGCN, "GWN": GraphWaveletNetwork}


class Raw(nn.Module):
    """Encoder output before its final ReLU (GCNLayer has none; dGCN and Spectral GCN end with relu(fc(.)))."""

    def __init__(self, enc: nn.Module):
        super().__init__()
        self.enc = enc

    def forward(self, x, adj):
        if isinstance(self.enc, GCNLayer):
            return self.enc(x, adj)
        box = []
        handle = self.enc.fc.register_forward_hook(lambda _m, _i, out: box.append(out))
        try:
            self.enc(x, adj)
        finally:
            handle.remove()
        return box[0]


class ControlEncoder(nn.Module):
    def __init__(self, enc: nn.Module, in_dim: int, out_dim: int, skip: bool, norm: bool):
        super().__init__()
        self.raw = Raw(enc)
        self.skip_connection = nn.Linear(in_dim, out_dim) if skip else None
        self.norm = nn.LayerNorm(out_dim) if norm else None

    def forward(self, x, adj):
        h = self.raw(x, adj)
        if self.skip_connection is not None:
            h = h + self.skip_connection(x)
        h = torch.relu(h)
        return self.norm(h) if self.norm is not None else h


VARIANTS = {"SkipOnly": (True, False), "NormOnly": (False, True), "Skip": (True, True)}
_build_gcn = fc.build_gcn


def build_gcn(name, input_dim, hidden_dim, dropout):
    base, _, kind = name.strip().partition("+")
    if kind in VARIANTS and base in ENCODERS:
        return ControlEncoder(ENCODERS[base](input_dim, hidden_dim), input_dim, hidden_dim, *VARIANTS[kind])
    return _build_gcn(name, input_dim, hidden_dim, dropout)


fc.build_gcn = build_gcn

# (dataset, H, gcn, tcn, arch, fusion, seed, tcn_dim, lr, amp)
RUNS = [
    ("pems-bay", 12, "GCN+Skip", "Default", "stack", "direct", 42, 64, 1e-3, True),
    ("pems-bay", 12, "GCN+Skip", "Default", "parallel", "fgm", 42, 64, 1e-3, True),
    ("metr-la", 12, "GCN", "Default", "stack", "direct", 42, 64, 1e-3, True),
    ("metr-la", 12, "GCN+Skip", "Default", "stack", "direct", 42, 64, 1e-3, True),
    ("metr-la", 12, "GCN", "Default", "parallel", "fgm", 42, 64, 1e-3, True),
    ("metr-la", 12, "GCN+Skip", "Default", "parallel", "fgm", 42, 64, 1e-3, True),
    ("pems-bay", 12, "GCN+SkipOnly", "Default", "stack", "direct", 42, 64, 1e-3, True),
    ("pems-bay", 12, "GCN+NormOnly", "Default", "stack", "direct", 42, 64, 1e-3, True),
    ("metr-la", 3, "GWN", "AMs", "stack", "direct", 42, 128, 5e-4, False),
    ("metr-la", 6, "GWN", "AMs", "stack", "direct", 42, 128, 5e-4, False),
    ("metr-la", 12, "GWN", "AMs", "stack", "direct", 42, 128, 5e-4, False),
    ("pems-bay", 12, "GCN+Skip", "Default", "stack", "direct", 123, 64, 1e-3, True),
    ("pems-bay", 12, "GCN+Skip", "Default", "stack", "direct", 456, 64, 1e-3, True),
    ("metr-la", 12, "GCN+SkipOnly", "Default", "stack", "direct", 42, 64, 1e-3, True),
    ("metr-la", 12, "GCN+NormOnly", "Default", "stack", "direct", 42, 64, 1e-3, True),
    ("metr-la", 12, "DiffusionGCN+SkipOnly", "Default", "stack", "direct", 42, 64, 1e-3, True),
    ("pems-bay", 12, "GWN+SkipOnly", "Default", "stack", "direct", 42, 64, 1e-3, True),
]


def self_test() -> None:
    torch.manual_seed(0)
    a = torch.rand(30, 30) + torch.eye(30)
    for base, cls in ENCODERS.items():
        e, x = cls(1, 64), torch.randn(4, 30, 1)
        raw = Raw(e)(x, a)
        ref = e(x, a)
        print(base, "raw vs encoder max diff", float(((raw if base == "GCN" else torch.relu(raw)) - ref).abs().max()))
    for g in sorted({r[2] for r in RUNS}):
        torch.manual_seed(0)
        m = fc.GcnTcnForecaster(fc.ForecasterConfig(num_nodes=30, history=12, horizon=12, gcn_variant=g, tcn_variant="Default",
                                                    architecture="stack", fusion_method="direct", parallel_spatial_mode="last",
                                                    gcn_dim=64, tcn_dim=64, dropout=0.0))
        y = m(torch.randn(4, 30, 12, 1), a)
        y.mean().backward()
        print(g, "output", tuple(y.shape), "parameters", sum(p.numel() for p in m.parameters()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=_ROOT / "data")
    ap.add_argument("--runs-root", type=Path, default=_ROOT / "runs_posthoc")
    ap.add_argument("--only", help="comma-separated run numbers from --list")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()
    if args.list:
        for i, r in enumerate(RUNS):
            print(i, *r)
        return
    if args.test:
        self_test()
        return
    import Fusion_ST_variants.train_ablation as t
    from Fusion_ST_variants.preprocessing.load_raw import load_raw
    from Fusion_ST_variants.preprocessing.scaler import compute_time_boundaries, fit_zscore
    from Fusion_ST_variants.utils.seed import enable_gpu_optimizations
    if not torch.cuda.is_available():
        raise SystemExit("a CUDA GPU is required for training")
    enable_gpu_optimizations()
    chosen = [int(i) for i in args.only.split(",")] if args.only else range(len(RUNS))
    data = {}
    for i in chosen:
        ds, q, g, tc, arch, fu, seed, width, lr, amp = RUNS[i]
        if ds not in data:
            speed, adj = load_raw(args.data_root / ds, ds)
            end, _ = compute_time_boundaries(speed.shape[0], 0.7, 0.2)
            sc = fit_zscore(speed, end)
            data[ds] = (sc.transform(speed), sc.mean, sc.std, adj)
        z, mean, std, adj = data[ds]
        root = args.runs_root / ("amp" if amp else "fp32_w128")
        start = time.time()
        print(f"[run {i}] {RUNS[i]}", flush=True)
        t.train_one_run(dataset_key=ds, speed_scaled=z, mean=mean, std=std, adj=adj, history=12, horizon=q, train_ratio=0.7, val_ratio=0.2,
                        gcn=g, tcn=tc, arch=arch, fusion=fu, seed=seed, batch_size=128, epochs=100, patience=15, lr=lr,
                        weight_decay=1e-5, dropout=0.2, gcn_dim=64, tcn_dim=width, amp=amp, results_csv=root / f"results__{ds}.csv",
                        runs_root=root, force=False, device=torch.device("cuda"), max_train_batches=0, max_eval_batches=0,
                        max_pred_batches=0, log_val_metrics=True, save_preds=False, parallel_spatial_mode="all_meanpool",
                        num_workers=0, exp_i=i, exp_n=len(RUNS))
        print(f"[run {i}] {(time.time() - start) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
