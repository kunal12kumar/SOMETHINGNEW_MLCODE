"""Compare two matching_results.tsv files on the test set, with examples.

Shows how many pairs each version adds or drops and prints samples, so a new
submission can be checked by eye against a known leaderboard result before
spending an upload.

Usage:
    python -m ber.compare_outputs --data-dir DATA --old output_v2/matching_results.tsv \
        --new output_v6/matching_results.tsv --n 20
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd


def read_pairs(path: Path) -> pd.DataFrame:
    m = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)
    m = m[m["matched_entity_ids"] != ""]
    m["cand_id"] = m["matched_entity_ids"].str.split(",")
    return m.explode("cand_id")[["source1_entity_id", "cand_id"]].rename(columns={"source1_entity_id": "s1_id"})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--old", type=Path, required=True)
    ap.add_argument("--new", type=Path, required=True)
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()

    old, new = read_pairs(args.old), read_pairs(args.new)
    j = old.merge(new, on=["s1_id", "cand_id"], how="outer", indicator=True)
    added, dropped = j[j["_merge"] == "right_only"], j[j["_merge"] == "left_only"]
    print(f"old {len(old):,} pairs, new {len(new):,} pairs; kept {int((j['_merge'] == 'both').sum()):,}, "
          f"added {len(added):,}, dropped {len(dropped):,}")

    sa = added.sample(min(args.n, len(added)), random_state=0)
    sd = dropped.sample(min(args.n, len(dropped)), random_state=0)
    ctx = pd.concat([new[new["s1_id"].isin(set(sa["s1_id"]))], old[old["s1_id"].isin(set(sd["s1_id"]))]])
    need = set(sa["s1_id"]) | set(sd["s1_id"]) | set(ctx["cand_id"]) | set(sd["cand_id"])
    raw = []
    for s in "123":
        path = args.data_dir / "test" / f"test_source{s}.tsv"
        for ch in pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE, chunksize=1_000_000):
            raw.append(ch[ch["entity_id"].isin(need)])
    raw = pd.concat(raw).set_index("entity_id")

    def line(e):
        r = raw.loc[e]
        return f"{e}: {r['business_name']!r} | {r['business_address']!r} [{r['country']}]"

    def show(title, sample, context):
        print(f"\n##### {title} #####")
        for sid, g in sample.groupby("s1_id", sort=False):
            print("S1 " + line(sid))
            marked = set(g["cand_id"])
            ids = list(dict.fromkeys(list(context[context["s1_id"] == sid]["cand_id"]) + list(marked)))
            for c in ids:
                print(("   >>> " if c in marked else "       ") + line(c))

    show("ADDED by new (>>>); other lines = new's matches", sa, new)
    show("DROPPED by new (>>>); other lines = old's matches", sd, old)


if __name__ == "__main__":
    main()
