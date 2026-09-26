"""Check normalisation on labelled match pairs: raw vs normalised agreement.

Usage:
    python -m ber.audit_normalize --data-dir ../student_resource/dataset --n-entities 20000
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz

from .io import read_ground_truth, read_source
from .normalize import normalize_record


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--n-entities", type=int, default=20_000)
    ap.add_argument("--show-worst", type=int, default=25)
    args = ap.parse_args()

    gt = read_ground_truth(args.data_dir)
    ids = gt["source1_entity_id"].drop_duplicates().sample(args.n_entities, random_state=0)
    pairs = gt[gt["source1_entity_id"].isin(ids)]

    recs = []
    for src in ("1", "2", "3"):
        df = read_source(args.data_dir, "train", src)
        want = set(ids) if src == "1" else set(pairs["matched_entity_id"])
        recs.append(df[df["entity_id"].isin(want)])
    recs = pd.concat(recs).set_index("entity_id")
    norm = pd.DataFrame(
        [normalize_record(n, a, c) for n, a, c in zip(recs.business_name, recs.business_address, recs.country)],
        index=recs.index,
    )
    recs = recs.join(norm)

    a = recs.loc[pairs["source1_entity_id"]].reset_index(drop=True)
    b = recs.loc[pairs["matched_entity_id"]].reset_index(drop=True)
    res = pd.DataFrame({
        "country": a["country"].values,
        "src": pairs["matched_entity_id"].str[:2].values,
        "name_raw": [fuzz.ratio(x.lower(), y.lower()) for x, y in zip(a.business_name, b.business_name)],
        "name_norm": [fuzz.ratio(x, y) for x, y in zip(a.name_norm, b.name_norm)],
        "name_core": [fuzz.ratio(x, y) for x, y in zip(a.name_core, b.name_core)],
        "core_exact": (a.name_core.values == b.name_core.values),
        "addr_raw": [fuzz.token_set_ratio(x.lower(), y.lower()) for x, y in zip(a.business_address, b.business_address)],
        "addr_norm": [fuzz.token_set_ratio(x, y) for x, y in zip(a.addr_norm, b.addr_norm)],
        "state_eq": [(x == y) if (x and y) else None for x, y in zip(a.addr_state, b.addr_state)],
        "city_eq": [(x == y) if (x and y) else None for x, y in zip(a.addr_city, b.addr_city)],
        "num_overlap": [bool(set(x.split()) & set(y.split())) if (x and y) else None for x, y in zip(a.addr_numbers, b.addr_numbers)],
    })
    pd.set_option("display.width", 200)
    print(f"{len(res):,} matched pairs from {args.n_entities:,} S1 entities\n")
    print(res.drop(columns=["country", "src"]).astype(float).mean().round(3).to_string(), "\n")
    print(res.groupby(["country", "src"]).mean(numeric_only=True).round(2), "\n")

    worst = res.nsmallest(args.show_worst, "name_core").index
    for i in worst:
        print(f"[{res.name_core[i]:.0f}] {a.business_name[i]!r:45} <> {b.business_name[i]!r}")
        print(f"      core: {a.name_core[i]!r:39} <> {b.name_core[i]!r}")


if __name__ == "__main__":
    main()
