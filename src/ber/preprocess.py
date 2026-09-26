"""Normalise every source file and cache it as parquet.

Files are streamed in blocks so memory stays flat on the 5M-row sources.

Usage:
    python -m ber.preprocess --data-dir ../student_resource/dataset --out-dir work/clean
"""
from __future__ import annotations

import argparse
import csv
import time
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .io import SOURCES, source_path
from .normalize import normalize_record


def _normalize_chunk(rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    return pd.DataFrame([normalize_record(n, a, c) for n, a, c in rows])


def normalize_frame(df: pd.DataFrame, pool: Pool | None, chunk: int = 20_000) -> pd.DataFrame:
    rows = list(zip(df["business_name"], df["business_address"], df["country"]))
    chunks = [rows[i : i + chunk] for i in range(0, len(rows), chunk)]
    parts = pool.map(_normalize_chunk, chunks) if pool else [_normalize_chunk(c) for c in chunks]
    norm = pd.concat(parts, ignore_index=True)
    norm.index = df.index
    return pd.concat([df, norm], axis=1)


def process_file(src_file: Path, out: Path, source: str, pool: Pool | None, block: int, limit: int | None) -> int:
    reader = pd.read_csv(
        src_file, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE,
        encoding="utf-8", chunksize=block, nrows=limit,
    )
    writer, n = None, 0
    tmp = out.with_suffix(".tmp")
    try:
        for df in reader:
            df["source"] = "S" + source
            table = pa.Table.from_pandas(normalize_frame(df, pool), preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(tmp, table.schema, compression="zstd")
            writer.write_table(table)
            n += len(df)
    finally:
        if writer is not None:
            writer.close()
    tmp.replace(out)
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--block", type=int, default=500_000, help="rows held in memory at once")
    ap.add_argument("--limit", type=int, default=None, help="rows per file, for quick tests")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pool = Pool(args.workers) if args.workers > 1 else None
    try:
        for split in args.splits:
            for src in SOURCES:
                out = args.out_dir / f"{split}_source{src}.parquet"
                if out.exists() and not args.overwrite:
                    print(f"skip {out.name} (exists)", flush=True)
                    continue
                t0 = time.time()
                n = process_file(source_path(args.data_dir, split, src), out, src, pool, args.block, args.limit)
                print(f"{out.name}: {n:,} rows in {time.time() - t0:.0f}s", flush=True)
    finally:
        if pool:
            pool.close()
            pool.join()


if __name__ == "__main__":
    main()
