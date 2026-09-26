"""Print test predictions for a country so errors can be read by eye.

For random S1 records of one country: the S1 record, the records we matched,
and the strongest candidates we did not match.

Usage:
    python -m ber.inspect_test --data-dir DATA --cands work/cands_test.parquet \
        --matching output/matching_results.tsv --country France --n 40
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd

from .io import read_source


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--cands", type=Path, required=True)
    ap.add_argument("--matching", type=Path, required=True)
    ap.add_argument("--country", default="France")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--show-unmatched", type=int, default=3)
    args = ap.parse_args()

    raw = pd.concat([read_source(args.data_dir, "test", s) for s in "123"]).set_index("entity_id")
    s1 = raw[raw["source"] == "S1"]
    m = pd.read_csv(args.matching, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)
    m = m.set_index("source1_entity_id")["matched_entity_ids"]

    by_country = s1["country"].value_counts()
    print("S1 per country:", by_country.to_dict())
    n_match = m.str.split(",").map(lambda x: len([i for i in x if i]))
    stats = pd.DataFrame({"n": n_match, "country": s1["country"].reindex(n_match.index)})
    print("matches per S1 (mean, share 0, share >=6) by country:")
    print(stats.groupby("country")["n"].agg(["mean", lambda x: (x == 0).mean(), lambda x: (x >= 6).mean()]).round(3))

    cols = ["s1_id", "cand_id", "both_score", "both_rank", "addr_score", "addr_rank"]
    ids = s1[s1["country"] == args.country].sample(args.n, random_state=1).index
    cands = pd.read_parquet(args.cands, columns=cols, filters=[("s1_id", "in", list(ids))])

    def line(eid):
        r = raw.loc[eid]
        return f"{eid}: {r['business_name']!r} | {r['business_address']!r}"

    for sid in ids:
        matched = [i for i in m.get(sid, "").split(",") if i]
        print("\n" + "=" * 100)
        print("S1  " + line(sid))
        for i in matched:
            print("  +  " + line(i))
        rest = cands[(cands["s1_id"] == sid) & ~cands["cand_id"].isin(matched)].sort_values("both_score", ascending=False)
        for _, c in rest.head(args.show_unmatched).iterrows():
            print(f"  -  [both {c.both_score:.2f}] " + line(c.cand_id))


if __name__ == "__main__":
    main()
