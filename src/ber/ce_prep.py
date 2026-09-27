"""Prepare pair lists and record texts for cross-encoder scoring.

  pairs    candidate parquet -> (s1_id, cand_id), optionally minus already-scored pairs
  records  pair parquet(s) + raw data -> (entity_id, text) for the records they use
  merge    combine several (s1_id, cand_id, p_ce) parquets into one

Usage:
    python -m ber.ce_prep pairs --cands V4/cands_test.parquet --exclude DRIVE/ce/test_scores_blend.parquet \
        --out DRIVE/ce/test_extra_pairs.parquet
    python -m ber.ce_prep records --data-dir DATA --split train --pairs DRIVE/ce/stack_pairs.parquet \
        --out DRIVE/ce/records_stack.parquet
    python -m ber.ce_prep merge --inputs A.parquet B.parquet --out C.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .export_ce import record_texts


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pairs")
    p.add_argument("--cands", type=Path, required=True)
    p.add_argument("--exclude", type=Path, default=None)
    p.add_argument("--out", type=Path, required=True)
    r = sub.add_parser("records")
    r.add_argument("--data-dir", type=Path, required=True)
    r.add_argument("--split", choices=["train", "test"], required=True)
    r.add_argument("--pairs", type=Path, nargs="+", required=True)
    r.add_argument("--out", type=Path, required=True)
    m = sub.add_parser("merge")
    m.add_argument("--inputs", type=Path, nargs="+", required=True)
    m.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    if args.cmd == "pairs":
        pairs = pd.read_parquet(args.cands, columns=["s1_id", "cand_id"]).drop_duplicates()
        n0 = len(pairs)
        if args.exclude:
            done = pd.read_parquet(args.exclude, columns=["s1_id", "cand_id"])
            pairs = pairs.merge(done, on=["s1_id", "cand_id"], how="left", indicator=True)
            pairs = pairs[pairs["_merge"] == "left_only"].drop(columns="_merge")
        pairs = pairs.sort_values(["s1_id", "cand_id"]).reset_index(drop=True)
        pairs.to_parquet(args.out, index=False)
        print(f"{len(pairs):,} pairs to score (of {n0:,}) -> {args.out}")
    elif args.cmd == "records":
        ids = set()
        for f in args.pairs:
            d = pd.read_parquet(f, columns=["s1_id", "cand_id"])
            ids |= set(d["s1_id"]) | set(d["cand_id"])
        rec = record_texts(args.data_dir, args.split, ids)
        rec.to_parquet(args.out, index=False, compression="zstd")
        print(f"{len(rec):,} records -> {args.out}")
    else:
        parts = [pd.read_parquet(f, columns=["s1_id", "cand_id", "p_ce"]) for f in args.inputs]
        out = pd.concat(parts, ignore_index=True).drop_duplicates(["s1_id", "cand_id"])
        out.to_parquet(args.out, index=False)
        print(f"{len(out):,} scored pairs -> {args.out}")


if __name__ == "__main__":
    main()
