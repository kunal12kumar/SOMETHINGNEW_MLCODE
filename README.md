# Business Entity Resolution

Pipeline: normalise → candidate generation → pair scoring → one-owner assignment → threshold.
Uses only the provided challenge files. No external data, APIs or lookups.

## Setup

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # Windows
# .venv/bin/python -m pip install -r requirements.txt     # Linux / Colab
```

All commands below run from this folder with `PYTHONPATH=src`.
`DATA` is the challenge `dataset/` folder (containing `train/` and `test/`).

## Steps

1. Normalise all source files to parquet (~10 min on 14 cores):
   ```bash
   python -m ber.preprocess --data-dir $DATA --out-dir work/clean --workers 14
   ```
2. Build the grouped validation split (10% of S1 entities):
   ```bash
   python -m ber.splits --clean-dir work/clean --out work/splits.parquet
   ```

Later steps (candidate generation, features, models, prediction) are added as they are built.

## Checks

```bash
python -m pytest -q tests
python -m ber.audit_normalize --data-dir $DATA --n-entities 20000
```

`audit_normalize` compares raw vs normalised similarity on labelled match pairs.
A normalisation rule is kept only if it improves agreement on this held-out check.
