"""Cross-encoder reranker: reads both records together and scores the pair.

Model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (Apache-2.0,
118M parameters), fine-tuned as a binary pair classifier. Its score is blended
with the LightGBM score; weight and threshold are tuned on validation macro F0.5.

Steps (each saves to --ce-dir and is skipped if its output exists):
  train   fine-tune on pairs_train.parquet            -> ce_model/
  valid   score validation pairs, tune the blend      -> blend.json, valid_scores.parquet
  test    score the test candidates and write outputs -> test_ce_chunks/, output files

Usage:
    python -m ber.cross_encoder --ce-dir DRIVE/ce --test-scores DRIVE/test_scores_v6.parquet \
        --candidate-file DRIVE/output_v6/candidate_pairs.tsv --out-dir DRIVE/output_ce
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from .metric import macro_f05

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # tokenising happens in DataLoader workers

MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MAX_LEN = 96


class PairData(Dataset):
    def __init__(self, a, b, y=None):
        self.a, self.b, self.y = a, b, y

    def __len__(self):
        return len(self.a)

    def __getitem__(self, i):
        return self.a[i], self.b[i], (self.y[i] if self.y is not None else 0.0)


def make_collate(tok):
    def collate(batch):
        a, b, y = zip(*batch)
        enc = tok(list(a), list(b), truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor(y, dtype=torch.float32)
        return enc
    return collate


def texts(pairs: pd.DataFrame, rec: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    return rec.reindex(pairs["s1_id"]).fillna("").to_numpy(), rec.reindex(pairs["cand_id"]).fillna("").to_numpy()


@torch.no_grad()
def score(model, tok, a, b, batch=1024, workers=8) -> np.ndarray:
    model.eval()
    dl = DataLoader(PairData(a, b), batch_size=batch, shuffle=False, num_workers=workers, collate_fn=make_collate(tok))
    out = []
    for enc in dl:
        enc.pop("labels")
        enc = {k: v.cuda(non_blocking=True) for k, v in enc.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(**enc).logits.float().squeeze(-1)
        out.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, np.float32)


def cmd_train(ce: Path, epochs: float, batch: int, lr: float) -> None:
    out = ce / "ce_model"
    if (out / "config.json").exists():
        print("skip train (ce_model exists)")
        return
    pairs = pd.read_parquet(ce / "pairs_train.parquet")
    rec = pd.read_parquet(ce / "records_train.parquet").drop_duplicates("entity_id").set_index("entity_id")["text"]
    a, b = texts(pairs, rec)
    y = pairs["label"].to_numpy(np.float32)
    # Show each pair in both orders so the score does not depend on which side is S1.
    a, b, y = np.concatenate([a, b]), np.concatenate([b, a]), np.concatenate([y, y])

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=1).cuda()
    dl = DataLoader(PairData(a, b, y), batch_size=batch, shuffle=True, num_workers=8,
                    collate_fn=make_collate(tok), drop_last=True)
    steps = int(len(dl) * epochs)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    model.train()
    t0, step, run = time.time(), 0, 0.0
    while step < steps:
        for enc in dl:
            labels = enc.pop("labels").cuda(non_blocking=True)
            enc = {k: v.cuda(non_blocking=True) for k, v in enc.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(**enc).logits.float().squeeze(-1)
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            run = 0.98 * run + 0.02 * loss.item() if step > 1 else loss.item()
            if step % 500 == 0 or step == steps:
                print(f"  step {step:,}/{steps:,} loss {run:.4f} ({time.time() - t0:.0f}s)", flush=True)
            if step >= steps:
                break
    tmp = ce / "ce_model_tmp"
    model.save_pretrained(tmp); tok.save_pretrained(tmp)
    if out.exists():
        shutil.rmtree(out)
    tmp.rename(out)
    print(f"trained in {time.time() - t0:.0f}s")


def decide(df: pd.DataFrame, col: str, t: float) -> pd.DataFrame:
    keep = (df[col].to_numpy() >= t) & (df[col] >= df.groupby("cand_id")[col].transform("max")).to_numpy()
    return df.loc[keep, ["s1_id", "cand_id"]]


def cmd_valid(ce: Path) -> dict:
    blend_file = ce / "blend.json"
    if blend_file.exists():
        cfg = json.loads(blend_file.read_text())
        print("blend (saved):", cfg)
        return cfg
    tok = AutoTokenizer.from_pretrained(ce / "ce_model")
    model = AutoModelForSequenceClassification.from_pretrained(ce / "ce_model").cuda()
    pv = pd.read_parquet(ce / "pairs_valid.parquet")
    rec = pd.read_parquet(ce / "records_train.parquet").drop_duplicates("entity_id").set_index("entity_id")["text"]
    a, b = texts(pv, rec)
    pv["p_ce"] = score(model, tok, a, b)
    pv.to_parquet(ce / "valid_scores.parquet", index=False)

    ids = pd.read_parquet(ce / "valid_s1.parquet")["s1_id"]
    gt = pd.read_parquet(ce / "gt_valid.parquet").rename(columns={"source1_entity_id": "s1_id", "matched_entity_id": "cand_id"})
    rows = []
    for w in (0.0, 0.3, 0.5, 0.7, 1.0):
        pv["pb"] = w * pv["p_ce"] + (1 - w) * pv["p_lgbm"]
        for t in (0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85):
            rows.append({"w_ce": w, "threshold": t, "f05": macro_f05(ids, decide(pv, "pb", t), gt)})
    res = pd.DataFrame(rows)
    print(res.pivot(index="threshold", columns="w_ce", values="f05").round(4).to_string())
    best = res.loc[res["f05"].idxmax()].to_dict()
    lgbm_only = res[res["w_ce"] == 0.0]["f05"].max()
    cfg = {"w_ce": float(best["w_ce"]), "threshold": float(best["threshold"]), "f05": float(best["f05"]),
           "lgbm_only_best": float(lgbm_only)}
    blend_file.write_text(json.dumps(cfg, indent=2))
    print("blend:", cfg)
    return cfg


def cmd_test(ce: Path, test_scores: Path, candidate_file: Path, out_dir: Path, threshold: float | None,
             chunk: int = 2_000_000) -> None:
    cfg = json.loads((ce / "blend.json").read_text())
    t = cfg["threshold"] if threshold is None else threshold
    tok = AutoTokenizer.from_pretrained(ce / "ce_model")
    model = AutoModelForSequenceClassification.from_pretrained(ce / "ce_model").cuda()
    sc = pd.read_parquet(test_scores)
    rec = pd.read_parquet(ce / "records_test.parquet").drop_duplicates("entity_id").set_index("entity_id")["text"]
    cdir = ce / "test_ce_chunks"
    cdir.mkdir(exist_ok=True)
    n = math.ceil(len(sc) / chunk)
    for i in range(n):
        f = cdir / f"chunk_{i:03d}.npy"
        if f.exists():
            continue
        t0 = time.time()
        part = sc.iloc[i * chunk:(i + 1) * chunk]
        a, b = texts(part, rec)
        np.save(f, score(model, tok, a, b).astype(np.float32))
        print(f"  test chunk {i + 1}/{n} ({time.time() - t0:.0f}s)", flush=True)
    sc["p_ce"] = np.concatenate([np.load(cdir / f"chunk_{i:03d}.npy") for i in range(n)])
    sc["pb"] = cfg["w_ce"] * sc["p_ce"] + (1 - cfg["w_ce"]) * sc["p"]
    sc[["s1_id", "cand_id", "p", "p_ce", "pb"]].to_parquet(ce / "test_scores_blend.parquet", index=False)

    matches = decide(sc, "pb", t)
    s1 = pd.read_parquet(ce / "test_s1.parquet")["entity_id"]
    lists = matches.groupby("s1_id")["cand_id"].agg(",".join)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "matching_results.tsv", "w", encoding="utf-8", newline="\n") as fh:
        fh.write("source1_entity_id\tmatched_entity_ids\n")
        for e in s1:
            fh.write(f"{e}\t{lists.get(e, '')}\n")
    shutil.copyfile(candidate_file, out_dir / "candidate_pairs.tsv")
    per = matches.groupby("s1_id").size()
    print(f"w_ce={cfg['w_ce']} threshold={t}: {len(matches):,} matches; "
          f"S1 with >=1 match {len(per) / len(s1):.3f}; mean per matched S1 {per.mean():.2f}")


def cmd_score(model_dir: Path, pairs_path: Path, records: list[Path], out: Path,
              part: int, n_parts: int, chunk: int = 2_000_000, max_chunks: int = 0) -> None:
    """Score any (s1_id, cand_id) parquet. Resumable; --part/--n-parts split work across sessions."""
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).cuda()
    pairs = pd.read_parquet(pairs_path, columns=["s1_id", "cand_id"])
    rec = pd.concat([pd.read_parquet(r) for r in records]).drop_duplicates("entity_id").set_index("entity_id")["text"]
    cdir = out.with_suffix(".chunks")
    cdir.mkdir(parents=True, exist_ok=True)
    n = math.ceil(len(pairs) / chunk)
    for i in range(part, min(n, max_chunks) if max_chunks else n, n_parts):
        f = cdir / f"chunk_{i:04d}.npy"
        if f.exists():
            continue
        t0 = time.time()
        p = pairs.iloc[i * chunk:(i + 1) * chunk]
        a, b = texts(p, rec)
        np.save(f, score(model, tok, a, b).astype(np.float32))
        print(f"  chunk {i + 1}/{n} ({time.time() - t0:.0f}s)", flush=True)
    done = [cdir / f"chunk_{i:04d}.npy" for i in range(n)]
    if all(f.exists() for f in done):
        pairs["p_ce"] = np.concatenate([np.load(f) for f in done])
        pairs.to_parquet(out, index=False)
        print(f"wrote {out} ({len(pairs):,} pairs)")
    else:
        print(f"{sum(f.exists() for f in done)}/{n} chunks done; run the other parts, then this again to merge")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ce-dir", type=Path, required=True)
    ap.add_argument("--steps", nargs="+", default=["train", "valid", "test"])
    ap.add_argument("--pairs", type=Path, help="score step: parquet with s1_id, cand_id")
    ap.add_argument("--records", type=Path, nargs="+", help="score step: record text parquets")
    ap.add_argument("--scores-out", type=Path, help="score step: output parquet")
    ap.add_argument("--model-dir", type=Path, default=None, help="score step: model folder (default ce-dir/ce_model)")
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--n-parts", type=int, default=1)
    ap.add_argument("--max-chunks", type=int, default=0, help="score step: stop after the first N chunks")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--test-scores", type=Path)
    ap.add_argument("--candidate-file", type=Path)
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--threshold", type=float, default=None)
    args = ap.parse_args()
    if "train" in args.steps:
        cmd_train(args.ce_dir, args.epochs, args.batch, args.lr)
    if "valid" in args.steps:
        cmd_valid(args.ce_dir)
    if "test" in args.steps:
        cmd_test(args.ce_dir, args.test_scores, args.candidate_file, args.out_dir, args.threshold)
    if "score" in args.steps:
        cmd_score(args.model_dir or args.ce_dir / "ce_model", args.pairs, args.records, args.scores_out,
                  args.part, args.n_parts, max_chunks=args.max_chunks)


if __name__ == "__main__":
    main()
