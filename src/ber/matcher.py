"""Two-stage LightGBM matcher, threshold tuning and final decisions.

Stage 1: pair features -> P(match). Out-of-fold on training entities.
Stage 2: pair features + within-S1 group features built from stage 1.
Decision: one-owner rule, then keep candidates above a threshold tuned for
macro F0.5 on validation entities.

Usage (train + evaluate):
    python -m ber.matcher train --data-dir DATA --clean-dir work/clean \
        --cands work/cands_train.parquet --model-dir work/model
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .features import REC_COLS, group_features, one_owner, pair_features, score_features, word_features
from .word_idf import load_lookup
from .io import read_ground_truth
from .metric import macro_f05

PARAMS = dict(
    objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
    verbose=-1, num_threads=0, seed=0,
)
N_FOLDS = 3
ROUNDS = 800


def load_records(clean_dir: Path, split: str, ids: set) -> pd.DataFrame:
    parts = []
    for s in ("1", "2", "3"):
        df = pd.read_parquet(clean_dir / f"{split}_source{s}.parquet", columns=REC_COLS)
        parts.append(df[df["entity_id"].isin(ids)])
    return pd.concat(parts).set_index("entity_id")


def build_features(cands: pd.DataFrame, recs: pd.DataFrame, chunk: int = 2_000_000) -> pd.DataFrame:
    """cands must be sorted by s1_id: chunks are cut only at S1 boundaries (support features need whole groups)."""
    s1 = cands["s1_id"].to_numpy()
    out, i = [], 0
    while i < len(cands):
        j = min(i + chunk, len(cands))
        while j < len(cands) and s1[j] == s1[j - 1]:
            j += 1
        t0 = time.time()
        out.append(pair_features(cands.iloc[i:j], recs))
        print(f"  features {j:,}/{len(cands):,} ({time.time() - t0:.0f}s)", flush=True)
        i = j
    return pd.concat(out)


def trim_candidates(cands: pd.DataFrame, addr_k: int, both_k: int) -> pd.DataFrame:
    """Keep a pair if it is within the addr top-K or the combined top-K of its source (0 = no trim)."""
    if not addr_k and not both_k:
        return cands
    keep = cands["addr_rank"].between(1, addr_k or 10**6) | cands["both_rank"].between(1, both_k or 10**6)
    out = cands[keep].copy()
    out["n_paths"] = ((out["addr_rank"].between(1, addr_k or 10**6)).astype(np.int8)
                      + (out["both_rank"].between(1, both_k or 10**6)).astype(np.int8))
    return out


def _fold_of(ids: pd.Series) -> np.ndarray:
    return (pd.util.hash_pandas_object(ids, index=False).to_numpy() % N_FOLDS).astype(int)


def _fit(X, y, rounds=ROUNDS, params=PARAMS):
    return lgb.train(params, lgb.Dataset(X, y, free_raw_data=True), num_boost_round=rounds)


def train_models(df: pd.DataFrame, feat_cols: list[str], rounds: int = ROUNDS, params: dict = PARAMS) -> dict:
    """df: training rows with features and label. Returns stage-1 fold models and stage-2 model."""
    folds = _fold_of(df["s1_id"])
    p1 = np.zeros(len(df), np.float32)
    stage1 = []
    for k in range(N_FOLDS):
        tr, te = folds != k, folds == k
        m = _fit(df.loc[tr, feat_cols], df.loc[tr, "label"], rounds, params)
        p1[te] = m.predict(df.loc[te, feat_cols])
        stage1.append(m)
        print(f"  stage1 fold {k} done", flush=True)
    df = df.assign(p1=p1)
    g = group_features(df, "p1")
    X2 = pd.concat([df[feat_cols + ["p1"]], g], axis=1)
    stage2 = _fit(X2, df["label"], rounds, params)
    return {"stage1": stage1, "stage2": stage2, "feat_cols": feat_cols, "stage2_cols": list(X2.columns)}


def predict(models: dict, df: pd.DataFrame) -> np.ndarray:
    fc = models["feat_cols"]
    p1 = np.mean([m.predict(df[fc]) for m in models["stage1"]], axis=0).astype(np.float32)
    df = df.assign(p1=p1)
    X2 = pd.concat([df[fc + ["p1"]], group_features(df, "p1")], axis=1)[models["stage2_cols"]]
    return models["stage2"].predict(X2).astype(np.float32)


def decide(df: pd.DataFrame, score: str, threshold: float, owner: bool = True) -> pd.DataFrame:
    keep = df[score].to_numpy() >= threshold
    if owner:
        keep &= one_owner(df, score)
    return df.loc[keep, ["s1_id", "cand_id"]]


def stress_copy(df: pd.DataFrame, score: str, mult: int = 2) -> pd.DataFrame:
    """Repeat hard negatives (non-matches scored > 0.3) so look-alikes are ~mult x as common, as on test."""
    hard = df[(df["label"] == 0) & (df[score] > 0.3)]
    reps = [hard.assign(cand_id=hard["cand_id"] + f"_dup{r}") for r in range(mult - 1)]
    return pd.concat([df] + reps, ignore_index=True)


def tune(df: pd.DataFrame, s1_ids, truth: pd.DataFrame, score: str) -> dict:
    """Threshold with the best stress score among those within 0.0005 of the best normal score."""
    st = stress_copy(df, score)
    rows = []
    for t in np.round(np.arange(0.30, 0.96, 0.05), 2):
        rows.append({"threshold": float(t),
                     "normal": macro_f05(s1_ids, decide(df, score, t, True), truth),
                     "stress": macro_f05(s1_ids, decide(st, score, t, True), truth)})
    res = pd.DataFrame(rows).set_index("threshold")
    print(res.round(4).to_string())
    ok = res[res["normal"] >= res["normal"].max() - 0.0005]
    t = float(ok["stress"].idxmax())
    return {"owner": True, "threshold": t, "macro_f05": float(res.loc[t, "normal"]),
            "stress_f05": float(res.loc[t, "stress"]), "best_normal": float(res["normal"].max())}


def add_extra_scores(cands: pd.DataFrame, paths) -> tuple[pd.DataFrame, list[str]]:
    """Attach external pair scores (s1_id, cand_id, p_ce) as columns ce0, ce1, ..."""
    names = []
    for i, path in enumerate(paths or []):
        name = f"ce{i}"
        e = pd.read_parquet(path, columns=["s1_id", "cand_id", "p_ce"]).drop_duplicates(["s1_id", "cand_id"])
        cands = cands.merge(e.rename(columns={"p_ce": name}), on=["s1_id", "cand_id"], how="left")
        missing = int(cands[name].isna().sum())
        print(f"extra score {name} from {path}: {missing:,} of {len(cands):,} pairs missing", flush=True)
        names.append(name)
    return cands, names


def cmd_train(args) -> None:
    cands = trim_candidates(pd.read_parquet(args.cands), args.addr_k, args.both_k)
    cands = cands.sort_values("s1_id", kind="stable").reset_index(drop=True)
    queries = pd.read_parquet(args.cands.with_name(args.cands.stem + "_queries.parquet"))
    gt = read_ground_truth(args.data_dir).rename(columns={"source1_entity_id": "s1_id", "matched_entity_id": "cand_id"})
    gt = gt[gt["s1_id"].isin(set(queries["s1_id"]))]
    cands = cands.merge(gt.assign(label=1), on=["s1_id", "cand_id"], how="left")
    cands["label"] = cands["label"].fillna(0).astype(np.int8)
    cands, extra = add_extra_scores(cands, args.extra_scores)
    if extra and args.require_extra:
        # Keep only S1 entities whose every candidate has the external score.
        bad = cands[extra].isna().any(axis=1).groupby(cands["s1_id"]).transform("any")
        dropped_valid = cands.loc[bad & (cands["role"] == "valid"), "s1_id"].nunique()
        assert dropped_valid == 0, f"{dropped_valid} validation S1 lack external scores"
        cands = cands[~bad].reset_index(drop=True)
        kept = set(cands["s1_id"])
        queries = queries[queries["s1_id"].isin(kept) | (queries["role"] == "valid")]
        gt = gt[gt["s1_id"].isin(kept) | gt["s1_id"].isin(set(queries.loc[queries["role"] == "valid", "s1_id"]))]
        print(f"kept {cands.loc[cands['role'] == 'train', 's1_id'].nunique():,} training S1 with external scores", flush=True)
    print(f"{len(cands):,} pairs, positives {cands['label'].mean():.3f}, "
          f"candidate recall {cands['label'].sum() / len(gt):.4f}", flush=True)

    if args.feature_cache and args.feature_cache.exists():
        feats = pd.read_parquet(args.feature_cache)
        print(f"loaded features from {args.feature_cache}", flush=True)
    else:
        ids = set(cands["s1_id"]) | set(cands["cand_id"])
        feats = build_features(cands, load_records(args.clean_dir, "train", ids))
        if args.feature_cache:
            feats.to_parquet(args.feature_cache, index=False)
    if args.word_idf:
        recs = load_records(args.clean_dir, "train", set(cands["s1_id"]) | set(cands["cand_id"]))
        feats = pd.concat([feats, word_features(cands, recs, load_lookup(args.word_idf))], axis=1)
        del recs
        print("added distinctive-word features", flush=True)
    for name in extra:
        feats = pd.concat([feats, score_features(cands, name)], axis=1)
    drop = tuple(p for p in (args.drop_features or "").split(",") if p)
    feat_cols = [c for c in feats.columns if not (drop and c.startswith(drop))]
    if drop:
        print(f"leaving out features starting with {drop}: {sorted(set(feats.columns) - set(feat_cols))}", flush=True)
    df = pd.concat([cands[["s1_id", "cand_id", "cand_source", "role", "label"]], feats[feat_cols]], axis=1)

    params = {**PARAMS, "learning_rate": args.lr, "num_leaves": args.leaves}
    tr, va = df[df["role"] == "train"].reset_index(drop=True), df[df["role"] == "valid"].reset_index(drop=True)
    t0 = time.time()
    models = train_models(tr, feat_cols, args.rounds, params)
    print(f"trained in {time.time() - t0:.0f}s", flush=True)

    va["p"] = predict(models, va)
    valid_ids = queries.loc[queries["role"] == "valid", "s1_id"]
    best = tune(va, valid_ids, gt[gt["s1_id"].isin(set(valid_ids))], "p")
    if args.threshold is not None:
        # Test has more look-alike distractors than validation, so a slightly higher
        # threshold than the validation optimum can be chosen deliberately.
        fixed = macro_f05(valid_ids, decide(va, "p", args.threshold, True), gt[gt["s1_id"].isin(set(valid_ids))])
        best = {"owner": True, "threshold": args.threshold, "macro_f05": fixed, "validation_best": best}
    print("best:", best)

    args.model_dir.mkdir(parents=True, exist_ok=True)
    for k, m in enumerate(models["stage1"]):
        m.save_model(str(args.model_dir / f"stage1_{k}.txt"))
    models["stage2"].save_model(str(args.model_dir / "stage2.txt"))
    (args.model_dir / "config.json").write_text(json.dumps(
        {"feat_cols": feat_cols, "stage2_cols": models["stage2_cols"], **best,
         "rounds": args.rounds, "learning_rate": args.lr, "num_leaves": args.leaves,
         "addr_k": args.addr_k, "both_k": args.both_k, "extra_scores": extra,
         "word_idf": bool(args.word_idf)}, indent=2))
    imp = pd.Series(models["stage2"].feature_importance("gain"), index=models["stage2_cols"])
    print((imp / imp.sum()).sort_values(ascending=False).head(20).round(4).to_string())


def load_models(model_dir: Path) -> tuple[dict, dict]:
    cfg = json.loads((model_dir / "config.json").read_text())
    models = {
        "stage1": [lgb.Booster(model_file=str(p)) for p in sorted(model_dir.glob("stage1_*.txt"))],
        "stage2": lgb.Booster(model_file=str(model_dir / "stage2.txt")),
        "feat_cols": cfg["feat_cols"], "stage2_cols": cfg["stage2_cols"],
    }
    return models, cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data-dir", type=Path, required=True)
    t.add_argument("--clean-dir", type=Path, required=True)
    t.add_argument("--cands", type=Path, required=True)
    t.add_argument("--model-dir", type=Path, required=True)
    t.add_argument("--rounds", type=int, default=ROUNDS)
    t.add_argument("--lr", type=float, default=PARAMS["learning_rate"])
    t.add_argument("--leaves", type=int, default=PARAMS["num_leaves"])
    t.add_argument("--feature-cache", type=Path, default=None, help="parquet to reuse features across runs")
    t.add_argument("--drop-features", default="", help="comma-separated feature name prefixes to leave out")
    t.add_argument("--word-idf", type=Path, default=None, help="train_token_idf.parquet: add distinctive-word features")
    t.add_argument("--require-extra", action="store_true",
                   help="train only on S1 entities whose candidates all have the --extra-scores")
    t.add_argument("--extra-scores", type=Path, nargs="*", default=None,
                   help="parquet(s) with s1_id, cand_id, p_ce added as features (must cover train and valid pairs)")
    t.add_argument("--threshold", type=float, default=None, help="fix the threshold instead of the validation best")
    t.add_argument("--addr-k", type=int, default=0, help="keep addr-path top-K per source (0 = all)")
    t.add_argument("--both-k", type=int, default=0, help="keep combined-path top-K per source (0 = all)")
    args = ap.parse_args()
    if args.cmd == "train":
        cmd_train(args)


if __name__ == "__main__":
    main()
