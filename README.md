# Business Entity Resolution

**Submitted: v10 = the matches v8-lite and v7 agree on, plus entities left empty filled where v7 and v6 agree (step 17).** v8-lite pipeline: normalise → state-blocked TF-IDF candidate search (25.7 per S1) →
a fine-tuned multilingual cross-encoder scores every candidate pair from the raw text →
a two-stage LightGBM uses the cross-encoder score (plus its rank and gap within the entity),
~40 similarity features and distinctive-word features → one-owner rule → threshold 0.75
(chosen on normal + stress validation). Validation macro F0.5 0.9791 (stress 0.9774).
The previous submission v7 (cross-encoder + LightGBM combined by a stacker, public LB 0.970) is
steps 10–12 below.

Only the provided challenge files are used. No external data, APIs, geocoding or lookups.
Pretrained weights downloaded: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
(Apache-2.0, 118M parameters), fine-tuned here. The fine-tuned cross-encoder ships in
`models/ce_model/`, the final LightGBM in `models/v8lite/`. The step-7 LightGBM (`models/v6/`) is
not shipped; step 7 recreates it.

Hardware used: 44-core / 172 GB RAM CPU machine for steps 1–9, one A100 GPU for steps 10–12
and 14, a 4-core / 31 GB machine for steps 15–16.

## Setup

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt    # Linux / Colab
# .venv/Scripts/python -m pip install -r requirements.txt  (Windows)
```

Run everything from this folder with `PYTHONPATH=src`.
`DATA` = the challenge `dataset/` folder (with `train/` and `test/`), `WORK` = a scratch folder.
Timings are for 44 CPU cores / 172 GB RAM (Colab TPU v6e-1 runtime used as a CPU machine).

## Steps 1–9: data, candidates and LightGBM (on its own this was submission v6)

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

Step 7 writes the model to `models/v6/`.
The shortlist trim (`--addr-k 5 --both-k 10`) is stored in the model config, so step 9
scores exactly the pairs written to `candidate_pairs.tsv` (25.7 per S1 on test).

## Steps 10–12: cross-encoder (GPU); with the stacker this was submission v7 (LB 0.970)

Uses the shortlist and LightGBM test scores from steps 8–9. A100: ~35 min training,
~110 min test scoring.

```bash
# 10. Pairs + raw record texts for the cross-encoder:
#     150k training-fold S1 (3M sampled pairs) + the same 30k validation S1,
#     with the v6 LightGBM score for every validation pair
python -m ber.run_candidates --split train --clean-dir $WORK/clean --splits $WORK/splits.parquet \
    --n-train 150000 --n-valid 30000 --threads 44 --neighbors $WORK/state_neighbors.json \
    --out $WORK/cands_train_ce.parquet
python -m ber.export_ce --data-dir $DATA --clean-dir $WORK/clean --cands $WORK/cands_train_ce.parquet \
    --model-dir models/v6 --out-dir $WORK/ce
# 11. Fine-tune (writes $WORK/ce/ce_model), score validation, score the 44.5M test pairs
#     (to reuse the shipped model instead: copy models/ce_model to $WORK/ce/ce_model; training is skipped)
python -m ber.cross_encoder --ce-dir $WORK/ce --steps train valid test \
    --test-scores $WORK/test_scores_v6.parquet --candidate-file output/candidate_pairs.tsv --out-dir $WORK/output_ce
# 12. Stacker (grouped 5-fold CV on validation), normal + stress validation, final decision;
#     writes output/matching_results.tsv (and copies candidate_pairs.tsv)
python -m ber.stack --ce-dir $WORK/ce --candidate-file output/candidate_pairs.tsv --out-dir output
```

## Steps 13–16: the submitted v8-lite

Needs the validation and test cross-encoder scores from step 11
(`$WORK/ce/valid_scores.parquet`, `$WORK/ce/test_scores_blend.parquet`).

```bash
# 13. Training entities never used by v6 or the cross-encoder (sample offset 1M) + the same
#     30k validation entities; word IDF tables for the distinctive-word features
python -m ber.run_candidates --split train --clean-dir $WORK/clean --splits $WORK/splits.parquet     --n-train 300000 --train-offset 1000000 --n-valid 30000 --threads 44     --neighbors $WORK/state_neighbors.json --out $WORK/cands_stack.parquet
python -m ber.word_idf --clean-dir $WORK/clean
# 14. Cross-encoder scores for training pairs. We scored the first 3 chunks (6M pairs of the
#     sorted pair list, ~109k entities) within our GPU budget; --max-chunks 3 reproduces that.
python -m ber.ce_prep pairs --cands $WORK/cands_stack.parquet --out $WORK/stack_pairs.parquet
python -m ber.ce_prep records --data-dir $DATA --split train --pairs $WORK/stack_pairs.parquet     --out $WORK/records_stack.parquet
python -m ber.cross_encoder --ce-dir $WORK/ce --steps score --pairs $WORK/stack_pairs.parquet     --records $WORK/records_stack.parquet --scores-out $WORK/ce_stack.parquet --max-chunks 3
python -m ber.ce_prep from-chunks --pairs $WORK/stack_pairs.parquet --chunks-dir $WORK/ce_stack.chunks     --out $WORK/ce_stack_partial.parquet
python -m ber.ce_prep merge --inputs $WORK/ce_stack_partial.parquet $WORK/ce/valid_scores.parquet     --out $WORK/ce_train.parquet
# 15. Train on the 99k training entities whose candidates all have a cross-encoder score (v7's
#     shortlist: addr top-5 + combined top-10 per source); prints normal + stress validation
python -m ber.matcher train --data-dir $DATA --clean-dir $WORK/clean --cands $WORK/cands_stack.parquet     --model-dir models/v8lite --extra-scores $WORK/ce_train.parquet --require-extra     --addr-k 5 --both-k 10 --drop-features sup_,addr_freq_     --word-idf $WORK/clean/train_token_idf.parquet --rounds 300 --lr 0.1
# 16. Test prediction -> output/matching_results.tsv, output/candidate_pairs.tsv
python -m ber.predict --clean-dir $WORK/clean --cands $WORK/cands_test.parquet --model-dir models/v8lite     --extra-scores $WORK/ce/test_scores_blend.parquet --word-idf $WORK/clean/test_token_idf.parquet     --out-dir output --batch-s1 150000
```

## Step 17: the submitted v10 (v9 + fill)

Keep a v8-lite match only if v7 (step 12) made it too; an entity v8-lite matched is never
left empty (this alone is v9, LB 0.97071). `--fill`: an entity that is still empty gets v7's matches that
v6 (step 9) also made, if no other entity owns them (v10). Rows follow v8-lite; `candidate_pairs.tsv`
is v8-lite's (the same shortlist as v7).

```bash
python -m ber.combine --a output_v7/matching_results.tsv --b output_v8lite/matching_results.tsv \n    --fill output_v6/matching_results.tsv --out-dir output
```

## Experiments that were not submitted

- Full v8: the 55/S1 shortlist (recall 98.2%) with cross-encoder features on all pairs; it needed
  ~68M more cross-encoder scores and could not be finished in time.

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
| `src/ber/export_ce.py`, `cross_encoder.py`, `stack.py` | Cross-encoder stage and score stacking |
| `src/ber/combine.py` | Agreement of two submissions (v9) |
| `src/ber/metric.py` | Official macro F0.5 |
| `src/ber/eval_candidates.py`, `analyze_errors.py`, `oracle.py`, `compare_outputs.py`, `inspect_test.py`, `audit_normalize.py`, `tune_blocking.py` | Analysis tools |
| `models/v8lite/` | Final LightGBM (with cross-encoder and word features) and chosen threshold |
| `models/ce_model/` | Fine-tuned cross-encoder |
| `tests/` | Unit tests for normalisation |
