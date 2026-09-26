"""Score test candidates and write the two submission files.

Usage:
    python -m ber.predict --clean-dir work/clean --cands work/cands_test.parquet \
        --model-dir work/model --out-dir output
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from .matcher import build_features, decide, load_models, load_records, predict


def write_id_lists(s1_ids: pd.Series, pairs: pd.DataFrame, id_col: str, path: Path) -> None:
    """One row per S1 entity (empty list allowed), IDs comma-joined, no duplicates."""
    lists = pairs.drop_duplicates().groupby("s1_id")["cand_id"].agg(",".join)
    out = pd.DataFrame({"source1_entity_id": s1_ids.values})
    out[id_col] = out["source1_entity_id"].map(lists).fillna("")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"source1_entity_id\t{id_col}\n")
        for a, b in zip(out["source1_entity_id"], out[id_col]):
            f.write(f"{a}\t{b}\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean-dir", type=Path, required=True)
    ap.add_argument("--cands", type=Path, required=True)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--threshold", type=float, default=None, help="override the tuned threshold")
    args = ap.parse_args()

    models, cfg = load_models(args.model_dir)
    threshold = cfg["threshold"] if args.threshold is None else args.threshold
    s1_ids = pd.read_parquet(args.clean_dir / "test_source1.parquet", columns=["entity_id"])["entity_id"]
    cands = pd.read_parquet(args.cands)
    print(f"{len(cands):,} candidate pairs for {len(s1_ids):,} test S1", flush=True)

    t0 = time.time()
    recs = load_records(args.clean_dir, "test", set(cands["s1_id"]) | set(cands["cand_id"]))
    feats = build_features(cands, recs)
    df = pd.concat([cands[["s1_id", "cand_id", "cand_source"]], feats], axis=1)
    df["p"] = predict(models, df)
    print(f"scored in {time.time() - t0:.0f}s", flush=True)

    matches = decide(df, "p", threshold, cfg.get("owner", True))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_id_lists(s1_ids, cands[["s1_id", "cand_id"]], "candidate_entity_ids", args.out_dir / "candidate_pairs.tsv")
    write_id_lists(s1_ids, matches, "matched_entity_ids", args.out_dir / "matching_results.tsv")
    df[["s1_id", "cand_id", "p"]].to_parquet(args.out_dir.parent / "work_test_scores.parquet", index=False)

    n_match = matches.groupby("s1_id").size()
    print(f"threshold={threshold} owner={cfg.get('owner', True)}: {len(matches):,} matches; "
          f"S1 with >=1 match {len(n_match) / len(s1_ids):.3f}; mean per matched S1 {n_match.mean():.2f}")
    country = pd.read_parquet(args.clean_dir / "test_source1.parquet", columns=["entity_id", "country_norm"])
    has = country["entity_id"].isin(set(matches["s1_id"]))
    print("share of S1 with a match, by country:", has.groupby(country["country_norm"]).mean().round(3).to_dict())


if __name__ == "__main__":
    main()
