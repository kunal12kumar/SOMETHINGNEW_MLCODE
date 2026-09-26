"""Score test candidates and write the two submission files.

Candidates are scored in batches of S1 entities (every candidate of an entity
stays in one batch, so within-entity group features are exact). The one-owner
rule needs all scores at once and runs after scoring on the small id/score table.

Usage:
    python -m ber.predict --clean-dir work/clean --cands work/cands_test.parquet \
        --model-dir work/model --out-dir output
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .features import pair_features
from .matcher import decide, load_models, load_records, predict


def write_id_lists(s1_ids: pd.Series, pairs: pd.DataFrame, id_col: str, path: Path) -> None:
    """One row per S1 entity (empty list allowed), IDs comma-joined, no duplicates."""
    lists = pairs.drop_duplicates().groupby("s1_id")["cand_id"].agg(",".join)
    out = pd.DataFrame({"source1_entity_id": s1_ids.values})
    out[id_col] = out["source1_entity_id"].map(lists).fillna("")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"source1_entity_id\t{id_col}\n")
        for a, b in zip(out["source1_entity_id"], out[id_col]):
            f.write(f"{a}\t{b}\n")


def score_in_batches(cands: pd.DataFrame, clean_dir: Path, models: dict, batch_s1: int) -> np.ndarray:
    cands = cands.sort_values("s1_id", kind="stable").reset_index(drop=True)
    s1_codes = pd.factorize(cands["s1_id"])[0]
    bounds = np.searchsorted(s1_codes, np.arange(0, s1_codes.max() + batch_s1 + 1, batch_s1))
    scores = np.zeros(len(cands), np.float32)
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        if lo >= hi:
            continue
        t0 = time.time()
        part = cands.iloc[lo:hi]
        recs = load_records(clean_dir, "test", set(part["s1_id"]) | set(part["cand_id"]))
        feats = pair_features(part, recs)
        df = pd.concat([part[["s1_id", "cand_id", "cand_source"]], feats], axis=1)
        scores[lo:hi] = predict(models, df)
        print(f"  scored {hi:,}/{len(cands):,} pairs ({time.time() - t0:.0f}s)", flush=True)
    return cands.assign(p=scores)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean-dir", type=Path, required=True)
    ap.add_argument("--cands", type=Path, required=True)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--batch-s1", type=int, default=150_000)
    ap.add_argument("--threshold", type=float, default=None, help="override the tuned threshold")
    ap.add_argument("--scores-out", type=Path, default=None, help="optional parquet of all pair scores")
    args = ap.parse_args()

    models, cfg = load_models(args.model_dir)
    threshold = cfg["threshold"] if args.threshold is None else args.threshold
    s1 = pd.read_parquet(args.clean_dir / "test_source1.parquet", columns=["entity_id", "country_norm"])
    cands = pd.read_parquet(args.cands)
    print(f"{len(cands):,} candidate pairs for {len(s1):,} test S1", flush=True)

    scored = score_in_batches(cands, args.clean_dir, models, args.batch_s1)
    if args.scores_out:
        scored[["s1_id", "cand_id", "p"]].to_parquet(args.scores_out, index=False)

    matches = decide(scored, "p", threshold, cfg.get("owner", True))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_id_lists(s1["entity_id"], cands[["s1_id", "cand_id"]], "candidate_entity_ids", args.out_dir / "candidate_pairs.tsv")
    write_id_lists(s1["entity_id"], matches, "matched_entity_ids", args.out_dir / "matching_results.tsv")

    n_match = matches.groupby("s1_id").size()
    print(f"threshold={threshold} owner={cfg.get('owner', True)}: {len(matches):,} matches; "
          f"S1 with >=1 match {len(n_match) / len(s1):.3f}; mean per matched S1 {n_match.mean():.2f}")
    has = s1["entity_id"].isin(set(matches["s1_id"]))
    print("share of S1 with a match, by country:", has.groupby(s1["country_norm"]).mean().round(3).to_dict())


if __name__ == "__main__":
    main()
