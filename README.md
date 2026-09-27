# Business Entity Resolution

Pipeline: normalise → state-blocked TF-IDF candidate search → pair features →
two-stage LightGBM → one-owner rule → threshold. An optional cross-encoder
(multilingual MiniLM, Apache-2.0) rescores the same candidates and is combined
with LightGBM.

Only the provided challenge files are used. No external data, APIs, geocoding or lookups.
Pretrained weights downloaded: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
(Apache-2.0, 118M parameters), used only by the optional cross-encoder stage.

## Setup

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt    # Linux / Colab
# .venv/Scripts/python -m pip install -r requirements.txt  (Windows)
```

Run everything from this folder with `PYTHONPATH=src`.
`DATA` = the challenge `dataset/` folder (with `train/` and `test/`), `WORK` = a scratch folder.
Timings are for 44 CPU cores / 172 GB RAM (Colab TPU v6e-1 runtime used as a CPU machine).

## End-to-end: LightGBM submission (v6)

```bash
# 1. Normalise all six source files to parquet                         (~4 min)
python -m ber.preprocess --data-dir $DATA --out-dir $WORK/clean --workers 44
# 2. Fill missing states from a city->state map learned from Source 1
python -m ber.fill_state --clean-dir $WORK/clean
# 3. Validation split (10% of S1 entities, grouped by entity)
python -m ber.splits --clean-dir $WORK/clean --out $WORK/splits.parquet
# 4. States often swapped between sources, learned from training labels
python -m ber.state_neighbors --data-dir $DATA --clean-dir $WORK/clean --out $WORK/state_neighbors.json
# 5. How common each core name / address location is (per split, per country)
python -m ber.name_freq --clean-dir $WORK/clean
# 6. Training candidates: 1M training + 30k validation S1                (~15 min)
python -m ber.run_candidates --split train --clean-dir $WORK/clean --splits $WORK/splits.parquet \
    --n-train 1000000 --n-valid 30000 --threads 44 --neighbors $WORK/state_neighbors.json \
    --out $WORK/cands_train_1m.parquet
# 7. Train the matcher on the smaller shortlist (addr top-5 + combined top-10 per source)  (~25 min)
python -m ber.matcher train --data-dir $DATA --clean-dir $WORK/clean --cands $WORK/cands_train_1m.parquet \
    --model-dir models/v6 --addr-k 5 --both-k 10 --drop-features sup_,addr_freq_ --threshold 0.75
# 8. Test candidates                                                     (~20 min)
python -m ber.run_candidates --split test --clean-dir $WORK/clean --threads 44 \
    --neighbors $WORK/state_neighbors.json --out $WORK/cands_test.parquet
# 9. Score, apply the one-owner rule and the threshold, write both files  (~15 min)
python -m ber.predict --clean-dir $WORK/clean --cands $WORK/cands_test.parquet --model-dir models/v6 \
    --out-dir output --batch-s1 400000 --scores-out $WORK/test_scores_v6.parquet
# official validator from the challenge's student_resource/utils folder
python $STUDENT_RESOURCE/utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir $DATA/test
```

The trained model used for the submission is in `models/v6/` (step 7 recreates it).
The shortlist trim (`--addr-k 5 --both-k 10`) is stored in the model config, so step 9
scores exactly the pairs written to `candidate_pairs.tsv` (25.7 per S1 on test).

## Optional: cross-encoder stage (GPU)

Runs on the shortlist and scores from above (A100: ~40 min training, ~40 min test scoring).

```bash
# export pairs + record texts for the cross-encoder
python -m ber.export_ce --data-dir $DATA --clean-dir $WORK/clean --cands $WORK/cands_train_1m.parquet \
    --model-dir models/v6 --out-dir $WORK/ce
# fine-tune, validate, score test
python -m ber.cross_encoder --ce-dir $WORK/ce --steps train valid test \
    --test-scores $WORK/test_scores_v6.parquet --candidate-file output/candidate_pairs.tsv --out-dir $WORK/output_ce
# combine LightGBM and cross-encoder (normal + stress validation), write the final files
python -m ber.stack --ce-dir $WORK/ce --candidate-file output/candidate_pairs.tsv --out-dir output_v7
```

## Checks and analysis tools

```bash
python -m pytest -q tests                                   # normalisation unit tests
python -m ber.audit_normalize --data-dir $DATA --n-entities 20000    # raw vs normalised agreement
python -m ber.eval_candidates --data-dir $DATA --clean-dir $WORK/clean --splits $WORK/splits.parquet
python -m ber.analyze_errors --data-dir $DATA --clean-dir $WORK/clean --cands $WORK/cands_train_1m.parquet --model-dir models/v6
python -m ber.compare_outputs --data-dir $DATA --old A/matching_results.tsv --new B/matching_results.tsv
```

`notebooks/colab_pipeline.ipynb` runs the early steps on Colab.

## Layout

| File | Purpose |
|---|---|
| `src/ber/lexicon.py` | Hand-written conventions: legal forms, address abbreviations, alias markers, postal state codes |
| `src/ber/normalize.py` | Name/address normalisation, Indic transliteration |
| `src/ber/preprocess.py` | Streams the six TSVs to normalised parquet |
| `src/ber/fill_state.py`, `state_neighbors.py`, `name_freq.py` | Data-driven state repair and frequency columns |
| `src/ber/splits.py` | Entity-grouped validation split |
| `src/ber/candidates.py`, `run_candidates.py` | State-blocked TF-IDF candidate generation |
| `src/ber/features.py` | Pair and within-entity group features |
| `src/ber/matcher.py` | Two-stage LightGBM, threshold tuning, shortlist trim |
| `src/ber/predict.py`, `rethreshold.py` | Test scoring and submission writing |
| `src/ber/export_ce.py`, `cross_encoder.py`, `stack.py` | Optional cross-encoder stage and score stacking |
| `src/ber/metric.py` | Official macro F0.5 |
| `src/ber/eval_candidates.py`, `analyze_errors.py`, `oracle.py`, `compare_outputs.py`, `inspect_test.py`, `audit_normalize.py`, `tune_blocking.py` | Analysis tools |
| `models/v6/` | Trained LightGBM models and chosen threshold |
| `tests/` | Unit tests for normalisation |
