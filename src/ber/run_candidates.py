"""Generate candidates for a set of S1 queries and save them.

Train: a sample of training-fold S1 entities (to fit the matcher) plus a sample
of validation-fold entities (to tune the threshold and score ourselves).
Test: every S1 entity.

Usage:
    python -m ber.run_candidates --split train --clean-dir work/clean --splits work/splits.parquet \
        --n-train 150000 --n-valid 30000 --out work/cands_train.parquet
    python -m ber.run_candidates --split test --clean-dir work/clean --out work/cands_test.parquet
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from .candidates import DEFAULT_PATHS, generate

COLS = ["entity_id", "country_norm", "addr_state", "name_norm", "addr_norm"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--clean-dir", type=Path, required=True)
    ap.add_argument("--splits", type=Path)
    ap.add_argument("--n-train", type=int, default=150_000)
    ap.add_argument("--n-valid", type=int, default=30_000)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    s1 = pd.read_parquet(args.clean_dir / f"{args.split}_source1.parquet", columns=COLS)
    if args.split == "train":
        sp = pd.read_parquet(args.splits)
        tr = sp.loc[~sp["is_valid"], "entity_id"].sample(args.n_train, random_state=0)
        va = sp.loc[sp["is_valid"], "entity_id"].sample(args.n_valid, random_state=0)
        s1 = s1[s1["entity_id"].isin(set(tr) | set(va))]
        roles = pd.Series("train", index=tr.values)
        roles[va.values] = "valid"
    pools = {s: pd.read_parquet(args.clean_dir / f"{args.split}_source{s[1]}.parquet", columns=COLS)
             for s in ("S2", "S3")}

    t0 = time.time()
    cands = generate(s1, pools, DEFAULT_PATHS, threads=args.threads)
    if args.split == "train":
        cands["role"] = cands["s1_id"].map(roles)
    cands.to_parquet(args.out, index=False)
    ids = pd.DataFrame({"s1_id": s1["entity_id"].values})
    if args.split == "train":
        ids["role"] = ids["s1_id"].map(roles)
    ids.to_parquet(args.out.with_name(args.out.stem + "_queries.parquet"), index=False)
    print(f"{len(cands):,} candidate pairs for {len(s1):,} S1 in {time.time() - t0:.0f}s "
          f"({len(cands) / len(s1):.1f} per S1)")


if __name__ == "__main__":
    main()
