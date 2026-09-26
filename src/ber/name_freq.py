"""Add how common each core name is within its split and country.

A rare name that matches exactly is strong evidence even when the address is
missing; a common name ("star trading") is not. Counted over Source 2 + Source 3
of the same split, so train and test are each counted on their own data.

Usage:
    python -m ber.name_freq --clean-dir work/clean
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean-dir", type=Path, required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    args = ap.parse_args()

    for split in args.splits:
        cols = ["country_norm", "name_core"]
        pool = pd.concat([pd.read_parquet(args.clean_dir / f"{split}_source{s}.parquet", columns=cols) for s in "23"])
        freq = pool.groupby(cols).size().rename("name_core_freq")
        for s in "123":
            path = args.clean_dir / f"{split}_source{s}.parquet"
            df = pd.read_parquet(path)
            df = df.drop(columns=["name_core_freq"], errors="ignore")
            df = df.join(freq, on=cols)
            df["name_core_freq"] = df["name_core_freq"].fillna(0).astype("int32")
            tmp = path.with_suffix(".tmp")
            df.to_parquet(tmp, index=False, compression="zstd")
            tmp.replace(path)
            print(f"{path.name}: median freq {df['name_core_freq'].median():.0f}, "
                  f"share unique (<=1) {(df['name_core_freq'] <= 1).mean():.3f}", flush=True)


if __name__ == "__main__":
    main()
