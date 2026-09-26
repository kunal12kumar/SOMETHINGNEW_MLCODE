"""Where is macro F0.5 lost? Score validation under 'oracle' fixes.

Each line removes one kind of error using the labels, to show how much score
it costs. This guides what to improve; it is not a model.

Usage:
    python -m ber.oracle --data-dir DATA --clean-dir work/clean --cands work/cands_train.parquet \
        --model-dir models/v2 --scores-out work/valid_scores_v2.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .features import pair_features
from .io import read_ground_truth
from .matcher import decide, load_models, load_records, predict
from .metric import macro_f05


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--clean-dir", type=Path, required=True)
    ap.add_argument("--cands", type=Path, required=True)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--scores-out", type=Path, required=True)
    args = ap.parse_args()

    models, cfg = load_models(args.model_dir)
    queries = pd.read_parquet(args.cands.with_name(args.cands.stem + "_queries.parquet"))
    valid_ids = queries.loc[queries["role"] == "valid", "s1_id"]
    gt = read_ground_truth(args.data_dir).rename(columns={"source1_entity_id": "s1_id", "matched_entity_id": "cand_id"})
    gt = gt[gt["s1_id"].isin(set(valid_ids))]

    if args.scores_out.exists():
        df = pd.read_parquet(args.scores_out)
    else:
        cands = pd.read_parquet(args.cands)
        cands = cands[cands["role"] == "valid"].reset_index(drop=True)
        recs = load_records(args.clean_dir, "train", set(cands["s1_id"]) | set(cands["cand_id"]))
        df = pd.concat([cands[["s1_id", "cand_id", "cand_source"]], pair_features(cands, recs)], axis=1)
        df["p"] = predict(models, df)
        df = df.merge(gt.assign(label=1), on=["s1_id", "cand_id"], how="left")
        df["label"] = df["label"].fillna(0).astype(np.int8)
        df.to_parquet(args.scores_out, index=False)

    t = cfg["threshold"]
    pred = decide(df, "p", t, cfg["owner"])
    lab = df.set_index(["s1_id", "cand_id"])["label"]
    is_true = lab.reindex(pd.MultiIndex.from_frame(pred)).to_numpy() == 1

    def score(p):
        return macro_f05(valid_ids, p, gt)

    rows = [("current model", score(pred))]
    rows.append(("no false matches", score(pred[is_true])))
    below = df[(df["label"] == 1) & (df["p"] < t)][["s1_id", "cand_id"]]
    rows.append(("+ all in-candidate misses recovered", score(pd.concat([pred[is_true], below]))))
    in_c = df[df["label"] == 1][["s1_id", "cand_id"]]
    rows.append(("perfect on candidates (recall ceiling)", score(in_c)))
    rows.append(("perfect overall", score(gt)))
    single = set(valid_ids) - set(gt["s1_id"])
    fp_single = pred[~is_true & pred["s1_id"].isin(single)]
    rows.append(("current, but singletons kept empty", score(pred[~pred["s1_id"].isin(single)])))
    print(f"false matches on singletons: {len(fp_single):,} pairs over {fp_single['s1_id'].nunique():,} entities")
    for name, s in rows:
        print(f"{name:42} {s:.4f}")

    # Threshold variants on the same scores
    for tt in (0.5, 0.6, 0.7, 0.8):
        print(f"threshold {tt}: {score(decide(df, 'p', tt, True)):.4f}")


if __name__ == "__main__":
    main()
