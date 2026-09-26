"""Learn which states are often swapped between sources, from training labels.

Example: an S1 record in "telangana" whose true S2 match says "andhra pradesh".
A query then also searches its neighbour states. Learned only from the
training ground truth.

Usage:
    python -m ber.state_neighbors --data-dir DATA --clean-dir work/clean --out work/state_neighbors.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .io import read_ground_truth

MIN_SHARE = 0.005  # of a state's labelled matches
MIN_PAIRS = 20


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--clean-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cols = ["entity_id", "country_norm", "addr_state"]
    st = pd.concat([pd.read_parquet(args.clean_dir / f"train_source{s}.parquet", columns=cols) for s in "123"])
    st = st.set_index("entity_id")
    gt = read_ground_truth(args.data_dir)
    a = st.loc[gt["source1_entity_id"].values]
    b = st.loc[gt["matched_entity_id"].values]
    d = pd.DataFrame({"country": a["country_norm"].values, "s1": a["addr_state"].values, "s2": b["addr_state"].values})
    d = d[(d["s1"] != "") & (d["s2"] != "")]
    counts = d.groupby(["country", "s1", "s2"]).size().rename("n").reset_index()
    counts["share"] = counts["n"] / counts.groupby(["country", "s1"])["n"].transform("sum")
    diff = counts[(counts["s1"] != counts["s2"]) & (counts["share"] >= MIN_SHARE) & (counts["n"] >= MIN_PAIRS)]

    neighbors: dict[str, list[str]] = {}
    for s1, s2 in zip(diff["s1"], diff["s2"]):
        for x, y in ((s1, s2), (s2, s1)):
            neighbors.setdefault(x, [])
            if y not in neighbors[x]:
                neighbors[x].append(y)
    args.out.write_text(json.dumps(neighbors, indent=1, sort_keys=True))
    agree = (d["s1"] == d["s2"]).mean()
    print(f"state agreement on labelled pairs: {agree:.4f}")
    print(diff.sort_values("n", ascending=False).head(20).to_string(index=False))


if __name__ == "__main__":
    main()
