"""Measure candidate recall on validation S1 entities.

recall@K = true matches found in the candidate set / all true matches.

Usage:
    python -m ber.eval_candidates --data-dir ../student_resource/dataset --clean-dir work/clean \
        --splits work/splits.parquet --n-queries 30000
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .candidates import DEFAULT_PATHS, generate
from .io import read_ground_truth

COLS = ["entity_id", "country_norm", "business_name", "business_address",
        "name_norm", "name_core", "name_alias", "addr_norm", "addr_city", "addr_numbers", "addr_state"]


def recall_table(cands: pd.DataFrame, truth: pd.DataFrame, paths, ks=(5, 10, 20)) -> pd.DataFrame:
    rows = []
    t = truth.merge(cands, left_on=["source1_entity_id", "matched_entity_id"],
                    right_on=["s1_id", "cand_id"], how="left")
    for k in ks:
        row = {"K": k}
        hit_any = False
        for p in paths:
            hit = t[f"{p.name}_rank"].between(1, k)
            row[p.name] = hit.mean()
            hit_any = hit_any | hit
        row["union"] = hit_any.mean()
        sub = cands[sum(cands[f"{p.name}_rank"].between(1, k) for p in paths) > 0]
        row["cands_per_s1"] = len(sub) / truth["source1_entity_id"].nunique()
        rows.append(row)
    return pd.DataFrame(rows).set_index("K")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--clean-dir", type=Path, required=True)
    ap.add_argument("--splits", type=Path, required=True)
    ap.add_argument("--n-queries", type=int, default=30_000)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--show-misses", type=int, default=30)
    args = ap.parse_args()

    split = pd.read_parquet(args.splits)
    valid_ids = split.loc[split["is_valid"], "entity_id"].sample(args.n_queries, random_state=0)
    s1 = pd.read_parquet(args.clean_dir / "train_source1.parquet", columns=COLS)
    queries = s1[s1["entity_id"].isin(valid_ids)]
    pools = {s: pd.read_parquet(args.clean_dir / f"train_source{s[1]}.parquet", columns=COLS) for s in ("S2", "S3")}

    cands = generate(queries, pools, DEFAULT_PATHS, threads=args.threads)
    if args.out:
        cands.to_parquet(args.out, index=False)

    gt = read_ground_truth(args.data_dir)
    truth = gt[gt["source1_entity_id"].isin(valid_ids)].copy()
    truth["src"] = truth["matched_entity_id"].str[:2]
    country = queries.set_index("entity_id")["country_norm"]
    truth["country"] = truth["source1_entity_id"].map(country)

    pd.set_option("display.width", 200)
    print(f"\n{len(queries):,} validation S1 queries, {len(truth):,} true pairs\n")
    print("Overall recall@K (K per source, per path):")
    print(recall_table(cands, truth, DEFAULT_PATHS).round(4), "\n")
    for (c, s), tr in truth.groupby(["country", "src"]):
        r = recall_table(cands, tr, DEFAULT_PATHS, ks=(20,))
        print(f"{c:>8} {s}: union@20={r['union'].iloc[0]:.4f}  " +
              "  ".join(f"{p.name}={r[p.name].iloc[0]:.4f}" for p in DEFAULT_PATHS))

    found = truth.merge(cands[["s1_id", "cand_id"]], left_on=["source1_entity_id", "matched_entity_id"],
                        right_on=["s1_id", "cand_id"], how="left")
    miss = found[found["cand_id"].isna()].sample(min(args.show_misses, found["cand_id"].isna().sum()), random_state=1)
    recs = pd.concat([queries] + list(pools.values())).set_index("entity_id")
    print(f"\nExamples of missed true matches:")
    for _, m in miss.iterrows():
        a, b = recs.loc[m.source1_entity_id], recs.loc[m.matched_entity_id]
        print(f"- {a.business_name!r} | {a.business_address!r}\n  {b.business_name!r} | {b.business_address!r}")


if __name__ == "__main__":
    main()
