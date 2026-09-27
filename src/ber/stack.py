"""Combine LightGBM and cross-encoder scores and choose the final decision rule.

Candidates compared on validation:
  lgbm      LightGBM probability alone (v6)
  ce        cross-encoder probability alone
  blend     w * ce + (1 - w) * lgbm
  stack     small LightGBM on both scores plus within-entity ranks and gaps,
            trained with 5-fold cross-validation grouped by S1 entity

Each is scored twice with the official macro F0.5 (one-owner rule applied):
  normal    the 30k validation entities as they are
  stress    the same, with hard negatives (non-matches either model scores
            above 0.3) repeated so there are ~2x as many look-alike
            distractors, as observed on the test set

The chosen rule must win on stress without losing on normal.

Usage (after `cross_encoder --steps valid` and `--steps test`):
    python -m ber.stack --ce-dir DRIVE/ce --candidate-file DRIVE/output_v6/candidate_pairs.tsv \
        --out-dir DRIVE/output_v7
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .metric import macro_f05

THRESHOLDS = (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9)
STACK_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=200,
                    feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=0)


def stack_features(df: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    a, b = df["p_lgbm"], df["p_ce"]
    f["p_lgbm"], f["p_ce"] = a, b
    f["prod"], f["diff"], f["absdiff"] = a * b, a - b, (a - b).abs()
    f["cand_is_s3"] = df["cand_id"].str.startswith("S3").astype(np.int8)
    for col in ("p_lgbm", "p_ce"):
        g = df.groupby("s1_id")[col]
        f[f"rank_{col}"] = g.rank(ascending=False, method="first")
        f[f"gap_{col}"] = g.transform("max") - df[col]
        f[f"n_above_{col}"] = (df[col] > 0.5).groupby(df["s1_id"]).transform("sum")
        src = df.groupby(["s1_id", f["cand_is_s3"]])[col].transform("max")
        f[f"gap_src_{col}"] = src - df[col]
    f["n_cands"] = df.groupby("s1_id")["p_lgbm"].transform("size")
    return f.astype(np.float32)


def decide(df: pd.DataFrame, col: str, t: float) -> pd.DataFrame:
    keep = (df[col].to_numpy() >= t) & (df[col] >= df.groupby("cand_id")[col].transform("max")).to_numpy()
    return df.loc[keep, ["s1_id", "cand_id"]]


def stress_copy(df: pd.DataFrame, mult: float = 2.0) -> pd.DataFrame:
    """Repeat hard negatives so look-alike distractors are ~mult x as common."""
    hard = df[(df["label"] == 0) & (df[["p_lgbm", "p_ce"]].max(axis=1) > 0.3)]
    reps = []
    for r in range(int(mult) - 1):
        d = hard.copy()
        d["cand_id"] = d["cand_id"] + f"_dup{r}"
        reps.append(d)
    return pd.concat([df] + reps, ignore_index=True)


def fit_stack(X: pd.DataFrame, y: np.ndarray, rounds: int = 300) -> lgb.Booster:
    return lgb.train(STACK_PARAMS, lgb.Dataset(X, y), num_boost_round=rounds)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ce-dir", type=Path, required=True)
    ap.add_argument("--candidate-file", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--force", default=None, help="method:threshold to use instead of the automatic choice")
    args = ap.parse_args()
    ce = args.ce_dir

    va = pd.read_parquet(ce / "valid_scores.parquet")
    ids = pd.read_parquet(ce / "valid_s1.parquet")["s1_id"]
    gt = pd.read_parquet(ce / "gt_valid.parquet")
    gt = gt.rename(columns={"source1_entity_id": "s1_id", "matched_entity_id": "cand_id"})[["s1_id", "cand_id"]]

    # Out-of-fold stacker predictions on validation (folds grouped by S1).
    X = stack_features(va)
    fold = (pd.util.hash_pandas_object(va["s1_id"], index=False).to_numpy() % 5).astype(int)
    va["p_stack"] = 0.0
    for k in range(5):
        m = fit_stack(X[fold != k], va.loc[fold != k, "label"].to_numpy())
        va.loc[fold == k, "p_stack"] = m.predict(X[fold == k])

    methods = {"lgbm": "p_lgbm", "ce": "p_ce", "stack": "p_stack"}
    for w in (0.3, 0.5, 0.7):
        va[f"p_blend{w}"] = w * va["p_ce"] + (1 - w) * va["p_lgbm"]
        methods[f"blend{w}"] = f"p_blend{w}"

    st = stress_copy(va)
    rows = []
    for name, col in methods.items():
        for t in THRESHOLDS:
            rows.append({"method": name, "threshold": t,
                         "normal": macro_f05(ids, decide(va, col, t), gt),
                         "stress": macro_f05(ids, decide(st, col, t), gt)})
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(res.pivot(index="threshold", columns="method", values="stress").round(4).to_string(), "\n  ^ stress (2x hard negatives)\n")
    print(res.pivot(index="threshold", columns="method", values="normal").round(4).to_string(), "\n  ^ normal\n")

    base = res[res["method"] == "lgbm"]
    base_normal, base_stress = base["normal"].max(), base.loc[base["threshold"] == 0.75, "stress"].iloc[0]
    ok = res[res["normal"] >= base_normal - 0.0005]
    best = ok.loc[ok["stress"].idxmax()]
    if args.force:
        name, t = args.force.split(":")[0], float(args.force.split(":")[-1])
        best = res[(res["method"] == name) & (np.isclose(res["threshold"], t))].iloc[0]
    choice = {"method": best["method"], "threshold": float(best["threshold"]),
              "normal": float(best["normal"]), "stress": float(best["stress"]),
              "v6_rule_stress": float(base_stress), "lgbm_best_normal": float(base_normal),
              "gain_stress_vs_v6": float(best["stress"] - base_stress)}
    print("choice:", json.dumps(choice, indent=2))
    (ce / "stack_choice.json").write_text(json.dumps(choice, indent=2))

    # Apply to test.
    te = pd.read_parquet(ce / "test_scores_blend.parquet").rename(columns={"p": "p_lgbm"})
    col = methods[choice["method"]]
    if choice["method"] == "stack":
        model = fit_stack(X, va["label"].to_numpy())
        te["p_stack"] = 0.0
        for s1_chunk in np.array_split(te["s1_id"].unique(), 20):
            mask = te["s1_id"].isin(set(s1_chunk)).to_numpy()
            te.loc[mask, "p_stack"] = model.predict(stack_features(te.loc[mask]))
    elif choice["method"].startswith("blend"):
        w = float(choice["method"][5:])
        te[col] = w * te["p_ce"] + (1 - w) * te["p_lgbm"]
    matches = decide(te, col, choice["threshold"])
    s1 = pd.read_parquet(ce / "test_s1.parquet")["entity_id"]
    lists = matches.groupby("s1_id")["cand_id"].agg(",".join)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "matching_results.tsv", "w", encoding="utf-8", newline="\n") as fh:
        fh.write("source1_entity_id\tmatched_entity_ids\n")
        for e in s1:
            fh.write(f"{e}\t{lists.get(e, '')}\n")
    shutil.copyfile(args.candidate_file, args.out_dir / "candidate_pairs.tsv")
    per = matches.groupby("s1_id").size()
    print(f"test: {len(matches):,} matches; S1 with >=1 match {len(per) / len(s1):.3f}; "
          f"mean per matched S1 {per.mean():.2f}; written to {args.out_dir}")


if __name__ == "__main__":
    main()
