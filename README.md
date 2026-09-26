# Business Entity Resolution

Pipeline: normalise → state-blocked TF-IDF candidate search → pair features →
two-stage LightGBM → one-owner rule → threshold tuned for macro F0.5.
Uses only the provided challenge files. No external data, APIs or lookups.

## Setup

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt    # Linux / Colab
# .venv/Scripts/python -m pip install -r requirements.txt  (Windows)
```

Run everything from this folder with `PYTHONPATH=src`.
`DATA` = the challenge `dataset/` folder (with `train/` and `test/`), `WORK` = a scratch folder.

## End-to-end

```bash
# 1. Normalise all six source files to parquet
python -m ber.preprocess --data-dir $DATA --out-dir $WORK/clean --workers 8
# 2. Fill missing states from a city->state map learned from Source 1
python -m ber.fill_state --clean-dir $WORK/clean
# 3. Validation split (10% of S1 entities, grouped by entity)
python -m ber.splits --clean-dir $WORK/clean --out $WORK/splits.parquet
# 4. States often swapped between sources, learned from training labels
python -m ber.state_neighbors --data-dir $DATA --clean-dir $WORK/clean --out $WORK/state_neighbors.json
# 4b. How common each core name is (per split, per country)
python -m ber.name_freq --clean-dir $WORK/clean
# 5. Training candidates (150k training + 30k validation S1)
python -m ber.run_candidates --split train --clean-dir $WORK/clean --splits $WORK/splits.parquet \
    --n-train 150000 --n-valid 30000 --neighbors $WORK/state_neighbors.json --out $WORK/cands_train.parquet
# 6. Train matcher, tune threshold on validation macro F0.5 (shipped model: models/v2)
python -m ber.matcher train --data-dir $DATA --clean-dir $WORK/clean \
    --cands $WORK/cands_train.parquet --model-dir $WORK/model
# 7. Test candidates and prediction -> output/matching_results.tsv, output/candidate_pairs.tsv
python -m ber.run_candidates --split test --clean-dir $WORK/clean \
    --neighbors $WORK/state_neighbors.json --out $WORK/cands_test.parquet
python -m ber.predict --clean-dir $WORK/clean --cands $WORK/cands_test.parquet \
    --model-dir models/v2 --out-dir output
```

`notebooks/colab_pipeline.ipynb` runs the same steps on Colab.

## Checks

```bash
python -m pytest -q tests
python -m ber.audit_normalize --data-dir $DATA --n-entities 20000      # raw vs normalised agreement
python -m ber.eval_candidates --data-dir $DATA --clean-dir $WORK/clean --splits $WORK/splits.parquet  # recall@K
```
