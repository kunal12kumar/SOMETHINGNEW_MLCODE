"""Export pairs and record texts for the cross-encoder (runs where the LightGBM run lives).

Writes to --out-dir:
  pairs_train.parquet    sampled training pairs (s1_id, cand_id, label)
  pairs_valid.parquet    all validation pairs with the LightGBM score p_lgbm
  valid_s1.parquet       every validation S1 id (including ones without candidates)
  gt_valid.parquet       validation ground-truth pairs
  records_train.parquet  entity_id -> text for records used above
  records_test.parquet   entity_id -> text for every test record
  test_s1.parquet        every test S1 id

Record text is the raw "name | address" string, so the model sees the original noise.

Usage:
    python -m ber.export_ce --data-dir DATA --clean-dir V4/clean --cands V4/cands_train_1m.parquet \
        --model-dir DRIVE/model_v6 --out-dir DRIVE/ce
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd

from .io import read_ground_truth
from .features import pair_features
from .matcher import load_models, load_records, predict, trim_candidates


def record_texts(data_dir: Path, split: str, ids: set | None) -> pd.DataFrame:
    parts = []
    for s in "123":
        path = data_dir / split / f"{split}_source{s}.tsv"
        for ch in pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE,
                              chunksize=2_000_000):
            if ids is not None:
                ch = ch[ch["entity_id"].isin(ids)]
            parts.append(pd.DataFrame({"entity_id": ch["entity_id"].values,
                                       "text": (ch["business_name"] + " | " + ch["business_address"]).values}))
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--clean-dir", type=Path, required=True)
    ap.add_argument("--cands", type=Path, required=True)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-train-pairs", type=int, default=3_000_000)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    models, cfg = load_models(args.model_dir)
    cands = trim_candidates(pd.read_parquet(args.cands), cfg.get("addr_k", 0), cfg.get("both_k", 0))
    cands = cands.sort_values("s1_id", kind="stable").reset_index(drop=True)
    queries = pd.read_parquet(args.cands.with_name(args.cands.stem + "_queries.parquet"))

    gt = read_ground_truth(args.data_dir).rename(columns={"source1_entity_id": "s1_id", "matched_entity_id": "cand_id"})
    gt = gt[gt["s1_id"].isin(set(queries["s1_id"]))]
    cands = cands.merge(gt.assign(label=1), on=["s1_id", "cand_id"], how="left")
    cands["label"] = cands["label"].fillna(0).astype(np.int8)

    # LightGBM score for validation pairs, computed here (validation is small).
    vc = cands[cands["role"] == "valid"].reset_index(drop=True)
    recs = load_records(args.clean_dir, "train", set(vc["s1_id"]) | set(vc["cand_id"]))
    df_valid = pd.concat([vc[["s1_id", "cand_id", "cand_source"]], pair_features(vc, recs)], axis=1)
    pv = vc[["s1_id", "cand_id", "label"]].copy()
    pv["p_lgbm"] = predict(models, df_valid)
    pv.to_parquet(args.out_dir / "pairs_valid.parquet", index=False)
    valid_ids = queries.loc[queries["role"] == "valid", ["s1_id"]]
    valid_ids.to_parquet(args.out_dir / "valid_s1.parquet", index=False)
    gt[gt["s1_id"].isin(set(valid_ids["s1_id"]))].to_parquet(args.out_dir / "gt_valid.parquet", index=False)
    print(f"valid: {len(pv):,} pairs, {len(valid_ids):,} S1", flush=True)

    tr = cands.loc[cands["role"] == "train", ["s1_id", "cand_id", "label"]]
    tr = tr.sample(min(args.n_train_pairs, len(tr)), random_state=0).reset_index(drop=True)
    tr.to_parquet(args.out_dir / "pairs_train.parquet", index=False)
    print(f"train sample: {len(tr):,} pairs, positives {tr['label'].mean():.3f}", flush=True)

    ids = set(tr["s1_id"]) | set(tr["cand_id"]) | set(pv["s1_id"]) | set(pv["cand_id"])
    rt = record_texts(args.data_dir, "train", ids)
    rt.to_parquet(args.out_dir / "records_train.parquet", index=False, compression="zstd")
    print(f"records_train: {len(rt):,}", flush=True)

    te = record_texts(args.data_dir, "test", None)
    te.to_parquet(args.out_dir / "records_test.parquet", index=False, compression="zstd")
    s1 = pd.read_parquet(args.clean_dir / "test_source1.parquet", columns=["entity_id"])
    s1.to_parquet(args.out_dir / "test_s1.parquet", index=False)
    print(f"records_test: {len(te):,}; test S1: {len(s1):,}", flush=True)


if __name__ == "__main__":
    main()
