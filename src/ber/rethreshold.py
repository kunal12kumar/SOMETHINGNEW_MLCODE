"""Write matching_results.tsv again from saved test scores with another threshold.

Usage:
    python -m ber.rethreshold --clean-dir work/clean --scores test_scores.parquet \
        --threshold 0.75 --out output/matching_results.tsv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .matcher import decide
from .predict import write_id_lists


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean-dir", type=Path, required=True)
    ap.add_argument("--scores", type=Path, required=True)
    ap.add_argument("--threshold", type=float, required=True)
    ap.add_argument("--no-owner", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    s1 = pd.read_parquet(args.clean_dir / "test_source1.parquet", columns=["entity_id", "country_norm"])
    scores = pd.read_parquet(args.scores)
    for t in (0.6, 0.65, 0.7, 0.75, 0.8, 0.85):
        n = len(decide(scores, "p", t, not args.no_owner))
        print(f"threshold {t}: {n:,} matches ({n / len(s1):.2f} per S1)")
    matches = decide(scores, "p", args.threshold, not args.no_owner)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_id_lists(s1["entity_id"], matches, "matched_entity_ids", args.out)
    print(f"wrote {args.out} with threshold {args.threshold}: {len(matches):,} matches")


if __name__ == "__main__":
    main()
