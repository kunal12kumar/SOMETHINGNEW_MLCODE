import csv
from pathlib import Path

import pandas as pd

SOURCES = ("1", "2", "3")


def source_path(data_dir: Path, split: str, source: str) -> Path:
    return Path(data_dir) / split / f"{split}_source{source}.tsv"


def read_source(data_dir: Path, split: str, source: str, nrows: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(
        source_path(data_dir, split, source),
        sep="\t",
        dtype=str,
        keep_default_na=False,
        quoting=csv.QUOTE_NONE,
        nrows=nrows,
        encoding="utf-8",
    )
    df["source"] = "S" + source
    return df


def read_ground_truth(data_dir: Path) -> pd.DataFrame:
    """Long format: one row per (source1_entity_id, matched_entity_id)."""
    gt = pd.read_csv(
        Path(data_dir) / "train" / "train_ground_truth.tsv",
        sep="\t",
        dtype=str,
        keep_default_na=False,
        quoting=csv.QUOTE_NONE,
    )
    gt["matched_entity_id"] = gt["matched_entity_ids"].str.split(",")
    long = gt.explode("matched_entity_id")[["source1_entity_id", "matched_entity_id"]]
    return long[long["matched_entity_id"].fillna("") != ""].reset_index(drop=True)
