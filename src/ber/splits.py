"""Deterministic train/validation split, grouped by Source 1 entity.

Each S1 entity and all of its matches land entirely in one fold, so no entity
leaks between training and validation. The fold comes from a hash of the ID,
so it is stable across runs and machines.

Usage:
    python -m ber.splits --clean-dir work/clean --out work/splits.parquet
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd

N_FOLDS = 10
VALID_FOLD = 0


def fold_of(entity_id: str, n_folds: int = N_FOLDS) -> int:
    return int(hashlib.md5(entity_id.encode()).hexdigest()[:8], 16) % n_folds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    s1 = pd.read_parquet(args.clean_dir / "train_source1.parquet", columns=["entity_id", "country"])
    s1["fold"] = [fold_of(e) for e in s1["entity_id"]]
    s1["is_valid"] = s1["fold"] == VALID_FOLD
    s1.to_parquet(args.out, index=False)
    print(s1.groupby(["country", "is_valid"]).size().unstack())


if __name__ == "__main__":
    main()
