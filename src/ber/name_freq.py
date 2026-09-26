"""Add how common each core name and each address location is within its split and country.

A rare name that matches exactly is strong evidence even when the address is
missing; a common name ("star trading") is not. Likewise a trade name at an
address used by one business is likely the same business, while a shared
office building is not. Counted over Source 2 + Source 3 of the same split, so
train and test are each counted on their own data.

Usage:
    python -m ber.name_freq --clean-dir work/clean
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def addr_key(df: pd.DataFrame) -> pd.Series:
    """Location key: the address numbers plus the city; empty when there are no numbers."""
    return (df["addr_numbers"] + "|" + df["addr_city"]).where(df["addr_numbers"] != "", "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean-dir", type=Path, required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    args = ap.parse_args()

    for split in args.splits:
        cols = ["country_norm", "name_core"]
        pool = pd.concat([pd.read_parquet(args.clean_dir / f"{split}_source{s}.parquet",
                                          columns=cols + ["addr_numbers", "addr_city"]) for s in "23"])
        freq = pool.groupby(cols).size().rename("name_core_freq")
        pool["addr_key"] = addr_key(pool)
        afreq = pool[pool["addr_key"] != ""].groupby(["country_norm", "addr_key"]).size().rename("addr_key_freq")
        del pool
        for s in "123":
            path = args.clean_dir / f"{split}_source{s}.parquet"
            df = pd.read_parquet(path)
            df = df.drop(columns=["name_core_freq", "addr_key_freq"], errors="ignore")
            df = df.join(freq, on=cols)
            df["name_core_freq"] = df["name_core_freq"].fillna(0).astype("int32")
            df["addr_key"] = addr_key(df)
            df = df.join(afreq, on=["country_norm", "addr_key"]).drop(columns=["addr_key"])
            df["addr_key_freq"] = df["addr_key_freq"].fillna(0).astype("int32")
            tmp = path.with_suffix(".tmp")
            df.to_parquet(tmp, index=False, compression="zstd")
            tmp.replace(path)
            print(f"{path.name}: median freq {df['name_core_freq'].median():.0f}, "
                  f"share unique (<=1) {(df['name_core_freq'] <= 1).mean():.3f}", flush=True)


if __name__ == "__main__":
    main()
