"""
RERUN RECORDED CONFIGURATIONS
=============================
The staged server scripts use one setting (tcn_dim 64, lr 1e-3). The 255 analysed runs do not: 56 of them used tcn_dim 128 and
lr 5e-4 (one of these also batch size 256). analysis/run_settings.csv lists the recorded setting of every run; this script trains
any of them with exactly that setting.

Run from the project root (parent of this repository):
  python Fusion_ST_variants/rerun_recorded.py --list --width 128
  python Fusion_ST_variants/rerun_recorded.py --match "metr-la__Q=12__gcn=GCN__tcn=Default" --data-root data
"""
from __future__ import annotations
import argparse
import csv
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for p in (str(_HERE), str(_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--settings", type=Path, default=_HERE / "analysis" / "run_settings.csv")
    ap.add_argument("--match", default="", help="substring of run_name")
    ap.add_argument("--width", type=int, choices=(64, 128))
    ap.add_argument("--data-root", type=Path, default=_ROOT / "data")
    ap.add_argument("--runs-root", type=Path, default=_ROOT / "runs_recorded")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    with open(args.settings, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if args.match in r["run_name"] and (args.width is None or int(r["tcn_dim"]) == args.width)]
    if args.list:
        for r in rows:
            print(r["run_name"], r["tcn_dim"], r["lr"], r["batch_size"], r["num_params"])
        print(len(rows), "runs")
        return
    if not args.match:
        raise SystemExit("give --match so that all 255 runs are not started by accident")
    import torch
    import Fusion_ST_variants.train_ablation as t
    from Fusion_ST_variants.preprocessing.load_raw import load_raw
    from Fusion_ST_variants.preprocessing.scaler import compute_time_boundaries, fit_zscore
    from Fusion_ST_variants.utils.seed import enable_gpu_optimizations
    if not torch.cuda.is_available():
        raise SystemExit("a CUDA GPU is required for training")
    enable_gpu_optimizations()
    data = {}
    for i, r in enumerate(rows):
        ds = r["dataset"]
        if ds not in data:
            speed, adj = load_raw(args.data_root / ds, ds)
            end, _ = compute_time_boundaries(speed.shape[0], 0.7, 0.2)
            sc = fit_zscore(speed, end)
            data[ds] = (sc.transform(speed), sc.mean, sc.std, adj)
        z, mean, std, adj = data[ds]
        t.train_one_run(dataset_key=ds, speed_scaled=z, mean=mean, std=std, adj=adj, history=12, horizon=int(r["horizon"]), train_ratio=0.7,
                        val_ratio=0.2, gcn=r["gcn"], tcn=r["tcn"], arch=r["arch"], fusion=r["fusion"], seed=int(r["seed"]),
                        batch_size=int(r["batch_size"]), epochs=100, patience=15, lr=float(r["lr"]), weight_decay=1e-5, dropout=0.2,
                        gcn_dim=int(r["gcn_dim"]), tcn_dim=int(r["tcn_dim"]), amp=True, results_csv=args.runs_root / f"results__{ds}.csv",
                        runs_root=args.runs_root, force=False, device=torch.device("cuda"), max_train_batches=0, max_eval_batches=0,
                        max_pred_batches=0, log_val_metrics=True, save_preds=False, parallel_spatial_mode="all_meanpool",
                        num_workers=0, exp_i=i, exp_n=len(rows))


if __name__ == "__main__":
    main()
