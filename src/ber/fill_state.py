"""Fill a missing state from the city, using a city -> state map learned from Source 1.

Source 1 addresses always carry both a city and a state. Where a Source 2/3
address has no recognisable state (e.g. French departments instead of regions),
we look up its place names in that map. Learned only from the provided files
of the same split; nothing external.

Usage:
    python -m ber.fill_state --clean-dir work/clean
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

MIN_COUNT = 3
MIN_PURITY = 0.9


def learn_city_state(s1: pd.DataFrame) -> dict[tuple[str, str], str]:
    d = s1[(s1["addr_city"] != "") & (s1["addr_state"] != "")]
    counts = d.groupby(["country_norm", "addr_city", "addr_state"]).size().rename("n").reset_index()
    tot = counts.groupby(["country_norm", "addr_city"])["n"].transform("sum")
    counts = counts[(counts["n"] >= MIN_COUNT) & (counts["n"] / tot >= MIN_PURITY)]
    return {(c, city): st for c, city, st in zip(counts["country_norm"], counts["addr_city"], counts["addr_state"])}


def infer_state(country: str, addr_norm: str, city_state: dict) -> str:
    toks = addr_norm.split()
    votes: dict[str, int] = {}
    for n in (3, 2, 1):
        for i in range(len(toks) - n + 1):
            st = city_state.get((country, " ".join(toks[i : i + n])))
            if st:
                votes[st] = votes.get(st, 0) + 1
    if len(votes) != 1:
        return ""  # none found, or conflicting places: leave unknown
    return next(iter(votes))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean-dir", type=Path, required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    args = ap.parse_args()

    for split in args.splits:
        s1 = pd.read_parquet(args.clean_dir / f"{split}_source1.parquet", columns=["country_norm", "addr_city", "addr_state"])
        city_state = learn_city_state(s1)
        print(f"{split}: learned {len(city_state):,} city -> state pairs")
        for s in ("2", "3"):
            path = args.clean_dir / f"{split}_source{s}.parquet"
            df = pd.read_parquet(path)
            if "addr_state_inferred" in df.columns:
                print(f"  skip {path.name} (already filled)")
                continue
            miss = df["addr_state"] == ""
            filled = [infer_state(c, a, city_state) for c, a in zip(df.loc[miss, "country_norm"], df.loc[miss, "addr_norm"])]
            df["addr_state_inferred"] = False
            df.loc[miss, "addr_state"] = filled
            df.loc[miss, "addr_state_inferred"] = [f != "" for f in filled]
            tmp = path.with_suffix(".tmp")
            df.to_parquet(tmp, index=False, compression="zstd")
            tmp.replace(path)
            still = (df["addr_state"] == "").groupby(df["country_norm"]).mean().round(3).to_dict()
            print(f"  {path.name}: filled {sum(f != '' for f in filled):,} of {int(miss.sum()):,}; still missing {still}")


if __name__ == "__main__":
    main()
