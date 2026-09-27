"""Keep only the matches two submissions agree on (v7 AND v8-lite).

A pair accepted by one model and rejected by the other is a borderline case; with F0.5
it is worth keeping only if it is right more than ~75% of the time, which such pairs rarely
are, especially on test with its denser look-alike businesses.
Exception: if dropping would leave the entity with no match at all, --b's matches are kept.
There the entity scores 0 or 1, so a match right more than half of the time is worth keeping.

Rows and ID order follow --b; candidate_pairs.tsv is copied from --b's folder.

Usage:
    python -m ber.combine --a output_v7/matching_results.tsv --b output_v8lite/matching_results.tsv \
        --out-dir output
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def read_matches(path: Path) -> dict[str, list[str]]:
    out = {}
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            s1, _, ids = line.rstrip("\n").partition("\t")
            out[s1] = ids.split(",") if ids else []
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=Path, required=True)
    ap.add_argument("--b", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--fill", type=Path, nargs="*", default=[],
                    help="for entities left with no match: add --a's matches that all these files also made")
    args = ap.parse_args()

    a = read_matches(args.a)
    with open(args.b, encoding="utf-8") as f:
        header = f.readline()
    b = read_matches(args.b)
    assert a.keys() == b.keys(), "the two files cover different S1 entities"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    kept = dropped = rows_kept = 0
    out = {}
    for s1, ids in b.items():
        both = set(a[s1])
        keep = [r for r in ids if r in both]
        if ids and not keep:
            keep = ids
            rows_kept += 1
        kept += len(keep); dropped += len(ids) - len(keep)
        out[s1] = keep
    filled = 0
    if args.fill:
        # an empty entity scores 0 or 1, so a match that 2+ models agree on is worth adding
        fills = [read_matches(p) for p in args.fill]
        owned = {r for ids in out.values() for r in ids}
        for s1, ids in out.items():
            if ids:
                continue
            add = [r for r in a[s1] if r not in owned and all(r in fl[s1] for fl in fills)]
            owned.update(add)
            out[s1] = add
            filled += bool(add)
    with open(args.out_dir / "matching_results.tsv", "w", encoding="utf-8", newline="\n") as f:
        f.write(header)
        for s1, ids in out.items():
            f.write(f"{s1}\t{','.join(ids)}\n")
    cand = args.b.parent / "candidate_pairs.tsv"
    if cand.exists():
        shutil.copy2(cand, args.out_dir / "candidate_pairs.tsv")
    print(f"{kept:,} matches kept, {dropped:,} dropped (only in --b); "
          f"{rows_kept:,} entities kept as in --b to avoid an empty row; {filled:,} empty entities filled "
          f"-> {args.out_dir}")


if __name__ == "__main__":
    main()
