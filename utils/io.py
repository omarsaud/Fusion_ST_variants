"""Simple IO helpers for experiment tracking.

This module is intentionally tiny:
- Consistent run naming
- Ensure output directory exists
- Save JSON configs/metrics
- Append rows to a CSV report
"""

import csv
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional


def run_name(
    dataset: str,
    horizon: int,
    gcn: str,
    tcn: str,
    arch: str,
    fusion: str,
    seed: int,
    parallel_spatial_mode: Optional[str] = None,
) -> str:
    """Return a stable run identifier string used as the run folder name."""
    ds = dataset.replace(" ", "").replace("_", "-")
    base = f"{ds}__Q={int(horizon)}__gcn={gcn}__tcn={tcn}__arch={arch}__fusion={fusion}"
    mode = None if parallel_spatial_mode is None else str(parallel_spatial_mode).strip()
    if mode and mode.lower() != "last":
        base = f"{base}__psm={mode}"
    return f"{base}__seed={int(seed)}"


def ensure_dir(p: Path) -> None:
    """Create directory `p` (and parents) if it does not exist."""
    os.makedirs(str(p), exist_ok=True)


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    """Write a JSON file with indentation."""
    ensure_dir(path.parent)
    with open(str(path), "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def append_csv(path: Path, row: Dict[str, Any], header: Optional[list] = None) -> None:
    """Append one row to a CSV file, writing header if the file is new."""
    ensure_dir(path.parent)
    if header is None:
        header = list(row.keys())

    header = list(header)
    for k in row.keys():
        if k not in header:
            header.append(k)

    def _fallback_path(p: Path) -> Path:
        base = p.with_suffix("")
        for i in range(1, 1000):
            cand = Path(f"{str(base)}__alt{i}{p.suffix}")
            if not cand.exists():
                return cand
        return Path(f"{str(base)}__alt{int(time.time())}{p.suffix}")

    if path.exists():
        with open(str(path), "r", newline="", encoding="utf-8") as rf:
            r = csv.DictReader(rf)
            existing_header = list(r.fieldnames or [])
            if existing_header and existing_header != header:
                new_header = list(existing_header)
                for k in header:
                    if k not in new_header:
                        new_header.append(k)
                rows = list(r)
                rows.append(row)

                try:
                    with open(str(path), "w", newline="", encoding="utf-8") as wf:
                        w = csv.DictWriter(wf, fieldnames=new_header, extrasaction="ignore")
                        w.writeheader()
                        for rr in rows:
                            w.writerow(rr)
                    return
                except PermissionError:
                    alt = _fallback_path(path)
                    with open(str(alt), "w", newline="", encoding="utf-8") as wf:
                        w = csv.DictWriter(wf, fieldnames=new_header, extrasaction="ignore")
                        w.writeheader()
                        for rr in rows:
                            w.writerow(rr)
                    print(f"[warn] could not write {path} (PermissionError). wrote {alt} instead.")
                    return

    exists = path.exists()
    try:
        with open(str(path), "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
            if not exists:
                w.writeheader()
            w.writerow(row)
    except PermissionError:
        alt = _fallback_path(path)
        with open(str(alt), "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
            w.writeheader()
            w.writerow(row)
        print(f"[warn] could not append {path} (PermissionError). wrote {alt} instead.")
