"""Rarity (IDF) of each word in core business names, per split and country.

Used by the distinctive-word features: a sibling business usually differs from
the S1 business by one rare word ("Infra" vs "Ventures", "Consultants" vs
"Textile"), while noise adds common words ("Services", "Center", "Group").
Counted over Source 2 + Source 3 of the same split only.

Usage:
    python -m ber.word_idf --clean-dir work/clean
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def build(clean_dir: Path, split: str) -> pd.DataFrame:
    pool = pd.concat([pd.read_parquet(clean_dir / f"{split}_source{s}.parquet", columns=["country_norm", "name_core"])
                      for s in "23"], ignore_index=True)
    n_docs = pool.groupby("country_norm").size().rename("n")
    toks = pool.assign(token=pool["name_core"].str.split()).explode("token").dropna(subset=["token"])
    toks = toks.reset_index().drop_duplicates(["index", "token"])
    df = toks.groupby(["country_norm", "token"]).size().rename("df").reset_index()
    df = df.join(n_docs, on="country_norm")
    df["idf"] = np.log((df["n"] + 1) / (df["df"] + 1)).astype(np.float32)
    return df[["country_norm", "token", "idf"]]


def load_lookup(path: Path) -> dict:
    t = pd.read_parquet(path)
    out = {}
    for c, g in t.groupby("country_norm"):
        out[c] = dict(zip(g["token"], g["idf"]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean-dir", type=Path, required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    args = ap.parse_args()
    for split in args.splits:
        t = build(args.clean_dir, split)
        out = args.clean_dir / f"{split}_token_idf.parquet"
        t.to_parquet(out, index=False)
        print(f"{out.name}: {len(t):,} tokens; median idf {t['idf'].median():.2f}")


if __name__ == "__main__":
    main()
