"""Validation error analysis for a trained matcher.

Usage:
    python -m ber.analyze_errors --data-dir DATA --clean-dir work/clean \
        --cands work/cands_train.parquet --model-dir models/v1
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .features import pair_features
from .io import read_ground_truth
from .matcher import decide, load_models, load_records, predict
from .metric import f05


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--clean-dir", type=Path, required=True)
    ap.add_argument("--cands", type=Path, required=True)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--examples", type=int, default=15)
    args = ap.parse_args()

    models, cfg = load_models(args.model_dir)
    cands = pd.read_parquet(args.cands)
    cands = cands[cands["role"] == "valid"].reset_index(drop=True)
    queries = pd.read_parquet(args.cands.with_name(args.cands.stem + "_queries.parquet"))
    valid_ids = queries.loc[queries["role"] == "valid", "s1_id"]

    recs = load_records(args.clean_dir, "train", set(cands["s1_id"]) | set(cands["cand_id"]) | set(valid_ids))
    feats = pair_features(cands, recs)
    df = pd.concat([cands[["s1_id", "cand_id", "cand_source"]], feats], axis=1)
    df["p"] = predict(models, df)

    gt = read_ground_truth(args.data_dir).rename(columns={"source1_entity_id": "s1_id", "matched_entity_id": "cand_id"})
    gt = gt[gt["s1_id"].isin(set(valid_ids))]
    pred = decide(df, "p", cfg["threshold"], cfg["owner"])

    P = pred.groupby("s1_id")["cand_id"].agg(set)
    T = gt.groupby("s1_id")["cand_id"].agg(set)
    per = pd.DataFrame({"s1_id": valid_ids.values})
    per["f"] = [f05(P.get(s, set()), T.get(s, set())) for s in per["s1_id"]]
    per["n_true"] = [len(T.get(s, set())) for s in per["s1_id"]]
    per["n_pred"] = [len(P.get(s, set())) for s in per["s1_id"]]
    s1_country = pd.read_parquet(args.clean_dir / "train_source1.parquet", columns=["entity_id", "country_norm"]).set_index("entity_id")["country_norm"]
    per["country"] = per["s1_id"].map(s1_country)

    print(f"macro F0.5 = {per['f'].mean():.4f}  (threshold {cfg['threshold']}, owner {cfg['owner']})")
    print(per.groupby("country")["f"].agg(["mean", "size"]).round(4).to_string())
    single = per["n_true"] == 0
    print(f"singletons: {single.mean():.3f} of S1, score {per.loc[single, 'f'].mean():.4f}; "
          f"non-singletons score {per.loc[~single, 'f'].mean():.4f}")

    tp = pred.merge(gt, on=["s1_id", "cand_id"])
    fp = pred.merge(gt, on=["s1_id", "cand_id"], how="left", indicator=True).query("_merge == 'left_only'")
    fn = gt.merge(pred, on=["s1_id", "cand_id"], how="left", indicator=True).query("_merge == 'left_only'")
    in_cands = fn.merge(df[["s1_id", "cand_id", "p"]], on=["s1_id", "cand_id"], how="left")
    print(f"pairs: TP {len(tp):,}  FP {len(fp):,}  FN {len(fn):,} "
          f"(of which not in candidates {in_cands['p'].isna().sum():,}, scored below threshold {in_cands['p'].notna().sum():,})")
    print(f"pair precision {len(tp) / max(len(pred), 1):.4f}, pair recall {len(tp) / len(gt):.4f}")

    raw = pd.concat([pd.read_parquet(args.clean_dir / f"train_source{s}.parquet", columns=["entity_id", "business_name", "business_address"])
                     for s in "123"]).set_index("entity_id")

    def show(title, rows):
        print(f"\n{title}")
        for _, r in rows.head(args.examples).iterrows():
            a, b = raw.loc[r.s1_id], raw.loc[r.cand_id]
            p = df.loc[(df.s1_id == r.s1_id) & (df.cand_id == r.cand_id), "p"]
            print(f"[{p.iloc[0] if len(p) else float('nan'):.2f}] {a.business_name!r} | {a.business_address!r}\n       {b.business_name!r} | {b.business_address!r}")

    show("False matches (predicted, not true):", fp.sample(min(len(fp), args.examples), random_state=0))
    show("Missed matches scored below threshold:", in_cands[in_cands["p"].notna()].sample(min(in_cands["p"].notna().sum(), args.examples), random_state=0))


if __name__ == "__main__":
    main()
